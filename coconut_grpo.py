import torch
torch.set_float32_matmul_precision('high')
import torch.nn.functional as F
from torch import Tensor
import contextlib
import time
import wandb
import datetime
from collections import defaultdict
import copy
import random
import Levenshtein  # Add this import for edit distance calculation


class GRPO:
    def __init__(
        self,
        model,
        ref_model,
        tokenizer=None,
        group_size=8,
        # micro_group_size=2,
        batch_size=1,
        max_iterations=1000,    
        dataset=None,
        reward_functions=None,
        log_wandb=False,
        dtype=None,
        lr=5e-6,
        weight_decay=0.0,
        beta=0.0,
        epsilon=0.1,
        noise_scale=3,
        # --- Parameters for latent reasoning ---
        max_continuous_tokens=1,
        latent_warmup_ratio=0.1,  # Keep num_thoughts=1 for first 10% of training by default
        forced_prefix="<answer>",  # Add forced tokens to insert <answer> tag
        stop_token=None,
        # --- Explorer parameters ---
        optimize_explorer=True,  # Whether to optimize Explorer parameters
        explorer_lr=1e-4,  # Learning rate for Explorer optimization
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.dataset = dataset.shuffle(seed=42)
        self.data_loader_iter = iter(self.dataset)
        self.group_size = group_size
        # self.micro_group_size = micro_group_size   
        self.batch_size = batch_size
        self.max_iterations = max_iterations
        self.dtype = dtype if dtype is not None else (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
        self.beta = beta
        self.epsilon = epsilon
        
        # Create separate parameter groups for model and Explorer
        model_params = [p for n, p in self.model.named_parameters() 
                         if not n.startswith('explorer.')]
        self.optimizer = torch.optim.AdamW(model_params, lr=lr, weight_decay=weight_decay, fused=True)
        
        # Add separate optimizer for Explorer if requested
        self.optimize_explorer = optimize_explorer
        if self.optimize_explorer and hasattr(self.model, 'explorer'):
            self.explorer_optimizer = torch.optim.AdamW(
                self.model.explorer.parameters(), 
                lr=explorer_lr, 
                weight_decay=0.0, 
                fused=True
            )
        
        assert reward_functions is not None, "Must pass reward_functions"
        self.reward_functions: list = reward_functions

        self.max_continuous_tokens = max_continuous_tokens
        self.continuous_tokens_rampup_steps = max_iterations
        self.latent_warmup_ratio = latent_warmup_ratio  # New parameter
        self.noise_scale = noise_scale

        self.model.to(self.device).to(dtype)
        
        # Ensure Explorer parameters have the same dtype as the model
        if hasattr(self.model, 'explorer'):
            for param in self.model.explorer.parameters():
                param.data = param.data.to(dtype)
        
        self.generate_config = copy.deepcopy(self.model.generation_config)
        self.stop_token = stop_token

        if forced_prefix is not None:
            # Tokenize and prepare forced tokens for <answer> tag
            self.forced_tokens = self.tokenizer(forced_prefix, add_special_tokens=False, return_tensors="pt").input_ids.to(self.device)

        if stop_token is not None:
            # Tokenize and prepare stop tokens for </answer> tag
            self.stop_token_ids = self.tokenizer(stop_token, add_special_tokens=False, return_tensors="pt").input_ids.to(self.device)


        self.using_lora = True if self.ref_model is None else False
        if beta > 0:
            if self.using_lora:
                self.ref_model = model
            self.ref_model.to(self.device).to(dtype)

        self.log_wandb = log_wandb
        if self.log_wandb:
            wandb.init(project="nanoGRPO")

        self.metrics = defaultdict(list)

    def get_per_token_logps(self, model, inputs) -> Tensor:
        """
        Compute log probabilities per token for latent-based inputs.
        
        - Uses 'inputs_embeds', 'attention_mask', and 'answer_token_ids' from the inputs dict
        """
        # Ensure inputs_embeds requires gradients for backpropagation to work
        if not inputs['inputs_embeds'].requires_grad:
            inputs['inputs_embeds'].requires_grad_(True)
        
        outputs = model(
            inputs_embeds=inputs['inputs_embeds'], 
            attention_mask=inputs['attention_mask'],
            use_cache=False
        )
        logits = outputs.logits
        token_ids = inputs['answer_token_ids']
        logps = F.log_softmax(logits, dim=-1)
        return torch.gather(logps, -1, token_ids.unsqueeze(-1)).squeeze(-1)

    def compute_loss(self, inputs, old_policy_log_probs, reward, mean_rewards, std_rewards, loss_mask) -> Tensor:
        policy_log_probs = self.get_per_token_logps(self.model, inputs)
        if old_policy_log_probs is None:
            old_policy_log_probs = policy_log_probs.detach()
        
        # Advantage calculation
        advantage: Tensor = (reward - mean_rewards) / (std_rewards + 1e-4)
        
        # Shape handling - ensure advantage has proper dimensions for broadcasting
        if advantage.dim() == 1:
            advantage = advantage.unsqueeze(-1)
        
        policy_ratio = torch.exp(policy_log_probs - old_policy_log_probs)
        
        loss1 = policy_ratio * advantage
        loss2 = torch.clamp(policy_ratio, 1 - self.epsilon, 1 + self.epsilon) * advantage
        loss = -torch.min(loss1, loss2)
        
        # Move everything to the same device
        loss_mask = loss_mask.to(loss.device)
        
        # Fix shape mismatch issue - reshape loss_mask to match the sequence dimension of loss
        if loss.dim() > loss_mask.dim():
            # If loss has more dimensions, we need to add dimensions to loss_mask
            while loss_mask.dim() < loss.dim():
                loss_mask = loss_mask.unsqueeze(0)
        
        # Now expand properly across the sequence dimension if needed
        if loss.size(-1) != loss_mask.size(-1):
            # Instead of trying to expand incompatible dimensions, 
            # create a proper-sized mask based on the loss shape
            seq_len = loss.size(-1)
            batch_size = loss.size(0) if loss.dim() > 1 else 1
            
            # Create a new mask of the proper size (all True)
            new_mask = torch.ones((batch_size, seq_len), dtype=torch.bool, device=loss.device)
            
            # Keep the original mask values for the overlapping part
            min_seq_len = min(seq_len, loss_mask.size(-1))
            if loss_mask.dim() == 1:
                new_mask[0, :min_seq_len] = loss_mask[:min_seq_len]
            else:
                new_mask[:, :min_seq_len] = loss_mask[:, :min_seq_len]
            
            loss_mask = new_mask.float()
        
        # Ensure loss_mask is float for multiplication
        if loss_mask.dtype != torch.float32:
            loss_mask = loss_mask.float()
        
        # Compute masked loss
        masked_loss = (loss * loss_mask).sum(dim=-1) / (loss_mask.sum(dim=-1) + 1e-6)
        
        if self.beta > 0:
            with (
                self.ref_model.disable_adapter()
                if self.using_lora  
                else contextlib.nullcontext()
            ):
                ref_policy_log_probs = self.get_per_token_logps(self.ref_model, inputs)
                
            # KL divergence calculation
            log_ratios = ref_policy_log_probs - policy_log_probs
            kld = torch.exp(log_ratios) - log_ratios - 1
            kld = (kld * loss_mask).sum(dim=-1) / (loss_mask.sum(dim=-1) + 1e-6)
            masked_loss += kld * self.beta
        
        return masked_loss.mean(), loss1.mean().item(), loss2.mean().item()

    def sample_batch(self):
        with torch.inference_mode():
            self.tokenizer.padding_side = "left"
            inputs_texts = []
            samples = []
            max_new_tokens = 0
            num_thoughts_sum = 0  # Track total number of thoughts
            
            for _ in range(self.batch_size):
                item = next(self.data_loader_iter)
                samples.append(item)
                prompt = item["prompt"]
                formatted = self.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
                
                c = self._sample_continuous_tokens()
                num_thoughts_sum += c  # Add to total
                max_new_tokens = max(max_new_tokens, c)
                formatted = formatted + f" <bot><num_thoughts={c}>"
                
                inputs_texts.append(formatted)

            # Calculate and track average number of thoughts
            avg_num_thoughts = num_thoughts_sum / self.batch_size
            self.metrics["avg_num_thoughts"].append(avg_num_thoughts)
            print(f"Average num_thoughts: {avg_num_thoughts:.2f}")

            max_new_tokens = max_new_tokens + 64
            encoded = self.tokenizer(inputs_texts, padding=True, return_tensors="pt")
            input_ids = encoded["input_ids"]
            attention_mask = encoded["attention_mask"]

            input_ids = torch.repeat_interleave(input_ids, self.group_size, dim=0)
            attention_mask = torch.repeat_interleave(attention_mask, self.group_size, dim=0)
            # Also repeat the input_texts for each group
            repeated_inputs_texts = [text for text in inputs_texts for _ in range(self.group_size)]
            # Fix: proper interleaving of samples (repeat each sample group_size times before moving to next)
            samples = [sample for sample in samples for _ in range(self.group_size)]

            forced_token_list = None
            if self.forced_tokens is not None:  
                # Create a list of forced tokens (one tensor per token)
                # Fix: properly expand the tensor with correct dimensions
                forced_token_list = [t.expand(input_ids.size(0), -1) for t in self.forced_tokens.split(1, dim=1)]

            start_time = time.time()
            
            outputs = self.model.generate(
                input_ids.to(self.device),
                attention_mask=attention_mask.to(self.device),
                max_new_tokens=max_new_tokens,
                input_text=repeated_inputs_texts,
                eos_token_id=self.tokenizer.eos_token_id,
                return_latent_states=True,  # Always return latent states
                noise_scale=self.noise_scale,
                forced_token_ids=forced_token_list,  # Add forced tokens to insert <answer> tag
            )
            end_time = time.time()
            print(f"Time for generation: {end_time - start_time} seconds")

        # Extract token_ids and create loss mask
        output_token_ids = outputs['answer_token_ids'].clone()
        
        # Create initial mask to filter out pad tokens
        loss_mask = output_token_ids != self.tokenizer.pad_token_id
        
        # Also mask out tokens after the EOS token
        eos_positions = (output_token_ids == self.tokenizer.eos_token_id).float()
        
        # Use cumulative sum to identify all positions after the first EOS token
        # When cumsum > 0 and not at the EOS position itself, it means we're after an EOS
        cumulative_eos = torch.cumsum(eos_positions, dim=-1)
        after_eos = (cumulative_eos > 0) & (output_token_ids != self.tokenizer.eos_token_id)
        
        # Update the loss mask to exclude tokens after EOS
        loss_mask = loss_mask & ~after_eos
        
        # Decode for rewards computation
        decoded_outputs = self.tokenizer.batch_decode(output_token_ids, skip_special_tokens=False)
        
        # Process decoded outputs to properly truncate at first EOS token
        processed_outputs = []
        for i, d in enumerate(decoded_outputs):
            # Remove the prompt prefix
            d = d.replace(inputs_texts[i // self.group_size], "")
            
            # Truncate at the first EOS token rather than just removing it
            eos_pos = d.find(self.tokenizer.eos_token)
            if eos_pos != -1:
                d = d[:eos_pos]  # Keep everything before the first EOS token
                
            # Still remove any pad tokens
            d = d.replace(self.tokenizer.pad_token, "")
            
            processed_outputs.append(d)
        
        # Calculate response diversity metrics
        diversity_score = self.calculate_response_diversity(processed_outputs)
        self.metrics["response_diversity"].append(diversity_score)
        # print(f"Response diversity: {diversity_score:.4f}")
        
        # Use the properly truncated outputs for reward computation
        rewards = self.compute_rewards(samples, processed_outputs)

        avg_decoded_length = loss_mask.sum(dim=-1).float() / loss_mask.shape[0]
        print(f"Average decoded length: {avg_decoded_length.mean().item()}")
        self.metrics["avg_decoded_length"].append(avg_decoded_length.mean().item())

        self.tokenizer.padding_side = "right"
        
        # Return batch result in unified format
        batch_result = {
            'rewards': rewards.clone().detach().to(dtype=torch.float32).to(self.device),
            'loss_mask': loss_mask.to(self.device),
            'inputs_embeds': outputs['input_embeds'].detach().clone().requires_grad_(True),
            'attention_mask': torch.ones_like(outputs['latent_mask'], dtype=torch.long),
            'answer_token_ids': output_token_ids,
            'mode': 'latent'
        }
        
        return batch_result

    def compute_rewards(self,samples, responses) -> list:
        rewards = [[[] for _ in range(self.batch_size)] for _ in range(len(self.reward_functions))]
        
        for idx, (sample, response) in enumerate(zip(samples, responses)):
            reward = 0
            for func_idx, func in enumerate(self.reward_functions):
                reward += func(sample, response)
                # print(f"{func.__name__} reward: {reward}")
                rewards[func_idx][idx % self.batch_size].append(reward)

        rewards = torch.tensor(rewards, dtype=self.dtype).to(self.device)

        # print(f"rewards: {rewards.shape}")
        for func_idx, func in enumerate(self.reward_functions):
            rwds = rewards[func_idx].mean(dim=-1)
            for r in rwds:
                self.metrics[f"reward_{func.__name__}"].append(r.item())

        prompt_lenghts = [[] for _ in range(self.batch_size)]
        for idx, sample in enumerate(samples):
            prompt_lenghts[idx % self.batch_size].append(len(sample["prompt"]))

        for idx, pl in enumerate(prompt_lenghts):
            self.metrics[f"prompt_length"].append(sum(pl)/len(pl))

        return rewards.sum(dim=0)
    
    def log_metrics(self):
        if self.log_wandb:
            if "idx" not in self.metrics or not self.metrics["idx"]:
                return  # No metrics to log yet
            
            idx = self.metrics["idx"][-1]-1
            metrics = {}
            for k, v in self.metrics.items():
                if not v:  # Skip empty lists
                    continue
                metrics[f"train/{k}"] = v[idx] if len(v) > idx else v[-1]
            
            wandb.log(metrics)

    def train(self, epochs=1, max_iterations=1000):
        idx = 0
        self.current_iteration = 0  # Add iteration tracking
        start_time = time.time()
        while idx < max_iterations:
            # Sample batch returns a dictionary with all necessary data
            self.current_iteration = idx  # Update current iteration before sampling
            batch_result = self.sample_batch()
            torch.cuda.empty_cache() 

            # Extract components from batch_result
            rewards = batch_result['rewards'].reshape(self.batch_size, self.group_size)
            loss_mask = batch_result['loss_mask']
            
            # Prepare inputs for latent mode
            batch_inputs = {
                'inputs_embeds': batch_result['inputs_embeds'],
                'attention_mask': batch_result['attention_mask'],
                'answer_token_ids': batch_result['answer_token_ids'],
                'mode': 'latent'
            }

            # offload to cpu to save vram
            batch_inputs = self._move_to_cpu(batch_inputs)
            rewards = rewards.cpu()
            loss_mask = loss_mask.cpu()
            torch.cuda.empty_cache()

            # Compute old policy log probs
            pi_old = []
            
            # For latent mode, compute log probs batch by batch
            for b_idx in range(self.batch_size):
                with torch.no_grad():
                    # Extract the batch slice for this particular batch
                    start_idx = b_idx * self.group_size
                    end_idx = start_idx + self.group_size
                    
                    # Move just what we need to GPU
                    gpu_inputs = {
                        'inputs_embeds': batch_inputs['inputs_embeds'][start_idx:end_idx].to(self.device),
                        'attention_mask': batch_inputs['attention_mask'][start_idx:end_idx].to(self.device),
                        'answer_token_ids': batch_inputs['answer_token_ids'][start_idx:end_idx].to(self.device),
                        'mode': 'latent'
                    }
                    b_old_policy_log_probs = self.get_per_token_logps(self.model, gpu_inputs).cpu()
                    torch.cuda.empty_cache()
                    pi_old.append(b_old_policy_log_probs)

            # Start the training loop
            for b_idx in range(self.batch_size):
                idx += 1
                
                # Get rewards for this batch
                reward = rewards[b_idx].to(self.device)
                mean_rewards = reward.mean(dim=-1).unsqueeze(-1)
                std_rewards = reward.std(dim=-1).unsqueeze(-1)
                
                # Get the slice for this batch index
                start_idx = b_idx * self.group_size
                end_idx = start_idx + self.group_size
                
                # Use the appropriate slice for latent mode
                b_inputs = {
                    'inputs_embeds': batch_inputs['inputs_embeds'][start_idx:end_idx].to(self.device),
                    'attention_mask': batch_inputs['attention_mask'][start_idx:end_idx].to(self.device),
                    'answer_token_ids': batch_inputs['answer_token_ids'][start_idx:end_idx].to(self.device),
                    'mode': 'latent'
                }
                    
                b_old_policy_log_probs = pi_old[b_idx].to(self.device)
                b_loss_mask = loss_mask[start_idx:end_idx].to(self.device)

                # Compute loss for the entire group
                loss, loss1, loss2 = self.compute_loss(
                    b_inputs,
                    b_old_policy_log_probs,
                    reward,
                    mean_rewards,
                    std_rewards,
                    b_loss_mask
                )
                
                # Both optimizers should be zero-graded before backward
                self.optimizer.zero_grad()
                if self.optimize_explorer and hasattr(self.model, 'explorer'):
                    self.explorer_optimizer.zero_grad()
                
                loss.backward()
                torch.cuda.empty_cache()    

                # Step both optimizers
                self.optimizer.step()
                if self.optimize_explorer and hasattr(self.model, 'explorer'):
                    self.explorer_optimizer.step()
                    
                    # Track Explorer statistics if wandb logging is enabled
                    if self.log_wandb:
                        with torch.no_grad():
                            explorer_mean_norm = self.model.explorer.mean.norm().item()
                            explorer_cov_norm = self.model.explorer.cholesky_factors.norm().item()
                            self.metrics["explorer_mean_norm"].append(explorer_mean_norm)
                            self.metrics["explorer_cov_norm"].append(explorer_cov_norm)

                print(f"{idx:04d} loss: {loss.item()} reward: {reward.mean()}")
                if self.log_wandb:
                    self.metrics["idx"].append(idx)
                    self.metrics["total_reward"].append(reward.mean().item())
                    self.metrics["loss"].append(loss.item())
                    self.metrics["ratio_loss"].append(loss1)
                    self.metrics["clipped_loss"].append(loss2)

                torch.cuda.empty_cache()
                
            print(f"iter {idx}  >>> reward: {rewards.mean()}")
            print(f"Total time: {str(datetime.timedelta(seconds=int(time.time() - start_time)))}")
            self.log_metrics()

    def _sample_continuous_tokens(self, iteration=None) -> int:
        """
        Sample the number of continuous tokens with a distribution that changes over training.
        
        During warmup period (first latent_warmup_ratio of training), always return 1.
        After warmup, at the beginning lower values are more likely.
        As training progresses, the distribution becomes more uniform.
        """
        # Use current iteration if provided, otherwise use a default value
        iteration = iteration if iteration is not None else getattr(self, 'current_iteration', 0)
        
        # During warmup period, always return 1
        warmup_steps = int(self.continuous_tokens_rampup_steps * self.latent_warmup_ratio)
        if iteration < warmup_steps:
            return 1
        
        # After warmup, use the existing scaling strategy
        # Calculate progress factor (0.0 at warmup end, 1.0 at end of training)
        effective_iteration = iteration - warmup_steps
        total_steps_after_warmup = self.continuous_tokens_rampup_steps - warmup_steps
        progress = min(1.0, effective_iteration / total_steps_after_warmup)
        
        # Calculate weighted probabilities for each possible token count
        # At start: heavily weighted toward 1
        # At end: approximately uniform
        weights = []
        for i in range(1, self.max_continuous_tokens + 1):
            # Linear interpolation between a distribution favoring low values and uniform
            weight = 1.0 - (progress * (i - 1) / self.max_continuous_tokens)
            weights.append(weight)
        
        # Normalize weights to form a probability distribution
        total = sum(weights)
        probs = [w / total for w in weights]
        
        # Sample using the calculated probability distribution
        return random.choices(range(1, self.max_continuous_tokens + 1), weights=probs, k=1)[0]

    def _move_to_cpu(self, inputs):
        """Helper to move input tensors to CPU"""
        if isinstance(inputs, dict):
            return {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        else:
            return inputs.cpu() if isinstance(inputs, torch.Tensor) else inputs

    def _move_to_device(self, inputs):
        """Helper to move input tensors to device"""
        if isinstance(inputs, dict):
            return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        else:
            return inputs.to(self.device) if isinstance(inputs, torch.Tensor) else inputs

    def calculate_response_diversity(self, responses):
        """
        Calculate diversity of responses within each group.
        
        Args:
            responses: List of all responses from all groups
            
        Returns:
            Average percentage of near-duplicates per group
        """
        # Group responses by batch item
        grouped_responses = [responses[i:i+self.group_size] for i in range(0, len(responses), self.group_size)]
        
        diversity_scores = []
        
        for group in grouped_responses:
            # Compare each response with every other response in the group
            total_similarity = 0
            num_comparisons = 0
            for i in range(len(group)):
                for j in range(i+1, len(group)):
                    # Calculate edit distance between responses
                    similarity = Levenshtein.ratio(group[i], group[j])
                    num_comparisons += 1
                    total_similarity += similarity

            # Calculate percentage of near-duplicates relative to total possible pairs
            if num_comparisons > 0:  # Avoid division by zero
                diversity_scores.append(1.0 - total_similarity / num_comparisons)  # Higher = more diverse
        
        avg_diversity = sum(diversity_scores) / len(diversity_scores) if diversity_scores else 0
        return avg_diversity
