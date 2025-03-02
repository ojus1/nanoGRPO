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

class GRPO:
    def __init__(
        self,
        model,
        ref_model,
        tokenizer=None,
        group_size=8,
        micro_group_size=2,
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
        # --- New parameters for latent reasoning ---
        max_continuous_tokens=1
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.dataset = dataset.shuffle(seed=42)
        self.data_loader_iter = iter(self.dataset)
        self.group_size = group_size
        self.micro_group_size = micro_group_size   
        self.batch_size = batch_size
        self.max_iterations = max_iterations
        self.dtype = dtype if dtype is not None else (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
        self.beta = beta
        self.epsilon = epsilon
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay, fused=True)
        assert reward_functions is not None, "Must pass reward_functions"
        self.reward_functions: list = reward_functions

        self.use_latent = True
        self.max_continuous_tokens = max_continuous_tokens
        self.continuous_tokens_rampup_steps = max_iterations

        self.model.to(self.device).to(dtype)
        # self.model = torch.compile(self.model, )
        # self.compiled_generate = torch.compile(self.model.generate, fullgraph=False, mode='default')
        self.generate_config = copy.deepcopy(self.model.generation_config)

        self.using_lora = True if self.ref_model is None else False
        if beta > 0:
            if self.using_lora:
                self.ref_model = model
            self.ref_model.to(self.device).to(dtype)

        self.log_wandb = log_wandb
        if self.log_wandb:
            wandb.init(project="nanoGRPO")

        self.metrics = defaultdict(list)


        # self.warmup_compile()
        # print("warmup done!")

    def warmup_compile(self):
        model_inputs = self.tokenizer(["Write a poem about the market crashing in summer"], return_tensors="pt")
        model_inputs = model_inputs.to(self.model.device)
        output_compiled = self.compiled_generate(**model_inputs, generation_config=self.generate_config)
        model_outputs = self.model(**model_inputs)

    def get_per_token_logps(self, model, inputs) -> Tensor:
        """
        Compute log probabilities per token for either token-based or latent-based inputs.
        
        For latent mode:
        - Uses 'inputs_embeds', 'attention_mask', and 'answer_token_ids' from the inputs dict
        
        For token mode:
        - Uses 'input_ids' directly
        """
        if isinstance(inputs, dict):
            # Latent mode
            if inputs.get('mode') == 'latent' or 'inputs_embeds' in inputs:
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
            else:
                # Token mode (inputs is a dict with input_ids)
                logits = model(input_ids=inputs['input_ids']).logits
                logits = logits[:, :-1, :]
                input_ids = inputs['input_ids'][:, 1:]
                logps = F.log_softmax(logits, dim=-1)
                return torch.gather(logps, -1, input_ids.unsqueeze(-1)).squeeze(-1)
        else:
            # Legacy token mode (inputs is a tensor)
            logits = model(input_ids=inputs).logits
            logits = logits[:, :-1, :]
            input_ids = inputs[:, 1:]
            logps = F.log_softmax(logits, dim=-1)
            return torch.gather(logps, -1, input_ids.unsqueeze(-1)).squeeze(-1)

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
        self.tokenizer.padding_side = "left"
        inputs_texts = []
        samples = []
        max_new_tokens = 0
        for _ in range(self.batch_size):
            item = next(self.data_loader_iter)
            samples.append(item)
            prompt = item["prompt"]
            formatted = self.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
            
            c = self._sample_continuous_tokens()
            max_new_tokens = max(max_new_tokens, c)
            formatted = formatted + f" <bot><num_thoughts={c}>"
            
            inputs_texts.append(formatted)

        max_new_tokens = max_new_tokens + 64
        encoded = self.tokenizer(inputs_texts, padding=True, return_tensors="pt")
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

        prompt_length = input_ids.shape[1]

        input_ids = torch.repeat_interleave(input_ids, self.group_size, dim=0)
        attention_mask = torch.repeat_interleave(attention_mask, self.group_size, dim=0)
        # Also repeat the input_texts for each group
        repeated_inputs_texts = [text for text in inputs_texts for _ in range(self.group_size)]
        samples = [sample for _ in range(self.group_size) for sample in samples]

        start_time = time.time()
        
        # Determine noise scale based on use_latent setting
        noise_scale = 3 if self.use_latent else 0.0
        
        outputs = self.model.generate(
            input_ids.to(self.device),
            attention_mask=attention_mask.to(self.device),
            max_new_tokens=max_new_tokens,
            input_text=repeated_inputs_texts,
            eos_token_id=self.tokenizer.eos_token_id,
            return_latent_states=self.use_latent,  # Only return latent states if we're in latent mode
            noise_scale=noise_scale,  # Add noise during exploration
        )
        end_time = time.time()
        print(f"Time for generation: {end_time - start_time} seconds")

        # Handle different output formats from generate
        if self.use_latent:
            # Latent mode returns a dictionary, extract answer_token_ids for further processing
            output_token_ids = outputs['answer_token_ids']
            # Store the full output dict for later processing with embeddings
            full_outputs = outputs
        else:
            # Standard token mode returns token IDs directly
            output_token_ids = outputs
            full_outputs = None

        loss_mask = torch.zeros_like(output_token_ids, dtype=torch.bool)

        # For latent mode, we don't need to trim the prompt since answer_token_ids already excludes the prompt
        if not self.use_latent:
            gen_tokens = output_token_ids[:, prompt_length:]
        else:
            gen_tokens = output_token_ids
        
        valid_gen_mask = gen_tokens != self.tokenizer.pad_token_id
        
        if not self.use_latent:
            loss_mask[:, prompt_length:] = valid_gen_mask
        else:
            loss_mask = valid_gen_mask

        # Decode for rewards computation
        decoded_outputs = self.tokenizer.batch_decode(output_token_ids, skip_special_tokens=False)
        decoded_outputs = [
            d
            .replace(inputs_texts[i // self.group_size], "")
            .replace(self.tokenizer.eos_token, "")
            .replace(self.tokenizer.pad_token, "")
            for i, d in enumerate(decoded_outputs)
        ]        
        rewards = self.compute_rewards(samples, decoded_outputs)

        avg_decoded_length = valid_gen_mask.sum(dim=-1).float() / valid_gen_mask.shape[0]
        print(f"Average decoded length: {avg_decoded_length.mean().item()}")
        self.metrics["avg_decoded_length"].append(avg_decoded_length.mean().item())

        self.tokenizer.padding_side = "right"
        
        # Always return a dictionary format for unified handling
        batch_result = {
            'rewards': rewards.clone().detach().to(dtype=torch.float32).to(self.device),
            'loss_mask': loss_mask.to(self.device)
        }
        
        if self.use_latent:
            # For latent mode, use the embeddings from the generation
            batch_result.update({
                'inputs_embeds': full_outputs['input_embeds'].detach().requires_grad_(True),
                'attention_mask': torch.ones_like(full_outputs['latent_mask'], dtype=torch.long),
                'answer_token_ids': output_token_ids,
                'mode': 'latent'
            })
        else:
            # For token mode, just pass the token IDs
            batch_result.update({
                'input_ids': output_token_ids,
                'mode': 'token'
            })
        
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
        start_time = time.time()
        while idx < max_iterations:
            # Sample batch returns a dictionary with all necessary data
            batch_result = self.sample_batch()
            torch.cuda.empty_cache() 

            # Extract components from batch_result
            mode = batch_result['mode']
            rewards = batch_result['rewards'].reshape(self.batch_size, self.group_size)
            loss_mask = batch_result['loss_mask'].reshape(self.batch_size, self.group_size, -1)
            
            # Prepare inputs based on mode
            if mode == 'latent':
                # Latent mode inputs
                batch_inputs = {
                    'inputs_embeds': batch_result['inputs_embeds'],
                    'attention_mask': batch_result['attention_mask'],
                    'answer_token_ids': batch_result['answer_token_ids'],
                    'mode': 'latent'
                }
            else:
                # Token mode inputs
                batch_inputs = {
                    'input_ids': batch_result['input_ids'].reshape(self.batch_size, self.group_size, -1),
                    'mode': 'token'
                }

            # offload to cpu to save vram
            batch_inputs = self._move_to_cpu(batch_inputs)
            rewards = rewards.cpu()
            loss_mask = loss_mask.cpu()
            torch.cuda.empty_cache()

            # Compute old policy log probs
            pi_old = []
            
            if mode == 'latent':
                # For latent mode, compute log probs batch by batch to match token mode structure
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
            else:
                # For token mode, process each batch separately
                for b_idx in range(self.batch_size):
                    with torch.no_grad():
                        b_input_ids = batch_inputs['input_ids'][b_idx].to(self.device)
                        b_old_policy_log_probs = self.get_per_token_logps(
                            self.model, {'input_ids': b_input_ids, 'mode': 'token'}
                        ).cpu()
                        torch.cuda.empty_cache()
                        pi_old.append(b_old_policy_log_probs)

            # Start the training loop
            for b_idx in range(self.batch_size):
                idx += 1
                
                # Get rewards for this batch
                reward = rewards[b_idx].to(self.device)
                mean_rewards = reward.mean(dim=-1).unsqueeze(-1)
                std_rewards = reward.std(dim=-1).unsqueeze(-1)
                
                # Prepare batch inputs
                if mode == 'latent':
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
                else:
                    # Get this batch's inputs for token mode
                    b_inputs = {
                        'input_ids': batch_inputs['input_ids'][b_idx].to(self.device),
                        'mode': 'token'
                    }
                    
                b_old_policy_log_probs = pi_old[b_idx]
                b_loss_mask = loss_mask[b_idx]

                # Split into micro-groups if needed
                if self.group_size == self.micro_group_size:
                    g_inputs = [self._move_to_device(b_inputs)]
                    g_old_policy_log_probs = [b_old_policy_log_probs.to(self.device)]
                    g_reward = [reward]
                    g_loss_mask = [b_loss_mask.to(self.device)]
                else:
                    # Split into micro groups
                    g_inputs = []
                    g_old_policy_log_probs = []
                    g_reward = []
                    g_loss_mask = []
                    
                    num_micro_groups = self.group_size // self.micro_group_size
                    
                    for mg_idx in range(num_micro_groups):
                        start_idx = mg_idx * self.micro_group_size
                        end_idx = start_idx + self.micro_group_size
                        
                        if mode == 'latent':
                            # For latent mode, slice the relevant parts
                            mg_inputs = {
                                'inputs_embeds': b_inputs['inputs_embeds'][start_idx:end_idx].to(self.device),
                                'attention_mask': b_inputs['attention_mask'][start_idx:end_idx].to(self.device),
                                'answer_token_ids': b_inputs['answer_token_ids'][start_idx:end_idx].to(self.device),
                                'mode': 'latent'
                            }
                        else:
                            # For token mode
                            mg_inputs = {
                                'input_ids': b_inputs['input_ids'][start_idx:end_idx].to(self.device),
                                'mode': 'token'
                            }
                            
                        g_inputs.append(mg_inputs)
                        g_old_policy_log_probs.append(b_old_policy_log_probs[start_idx:end_idx].to(self.device))
                        g_reward.append(reward[start_idx:end_idx])
                        g_loss_mask.append(b_loss_mask[start_idx:end_idx].to(self.device))
                    
                group_losses = []
                group_losses1 = []
                group_losses2 = []

                for inputs, old_policy_log_probs, reward, loss_mask in zip(
                    g_inputs, g_old_policy_log_probs, g_reward, g_loss_mask
                ):
                    loss, loss1, loss2 = self.compute_loss(
                        inputs,
                        old_policy_log_probs,
                        reward,
                        mean_rewards,
                        std_rewards,
                        loss_mask
                    )
                    group_losses.append(loss.item())
                    group_losses1.append(loss1)
                    group_losses2.append(loss2)
                    loss.backward()
                    torch.cuda.empty_cache()    

                self.optimizer.step()
                self.optimizer.zero_grad()

                print(f"{idx:04d} loss: {sum(group_losses)/len(group_losses)} reward: {reward.mean()}")
                if self.log_wandb:
                    self.metrics["idx"].append(idx)
                    self.metrics["total_reward"].append(reward.mean().item())
                    self.metrics["loss"].append(sum(group_losses)/len(group_losses))
                    self.metrics["ratio_loss"].append(sum(group_losses1)/len(group_losses1))
                    self.metrics["clipped_loss"].append(sum(group_losses2)/len(group_losses2))

                torch.cuda.empty_cache()
                
            print(f"iter {idx}  >>> reward: {rewards.mean()}")
            print(f"Total time: {str(datetime.timedelta(seconds=int(time.time() - start_time)))}")
            self.log_metrics()

    def _sample_continuous_tokens(self) -> int:
        """
        Sample the number of continuous tokens uniformly from 1 to max_continuous_tokens.
        (No ramp-up for max continuous tokens is used.)
        """
        return random.randint(1, self.max_continuous_tokens)

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
