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
        epsilon=0.1
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
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr,weight_decay=weight_decay, fused=True)
        assert reward_functions is not None, "Must pass reward_functions"
        self.reward_functions: list = reward_functions

        self.model.to(self.device).to(dtype)
        # self.model = torch.compile(self.model, )
        # self.compiled_generate = torch.compile(self.model.generate, fullgraph=False, mode='default')
        self.generate_config = copy.deepcopy(self.model.generation_config)

        self.using_lora = True if self.ref_model is None else False
        if self.using_lora:
            self.ref_model = model

        self.log_wandb = log_wandb
        if self.log_wandb:
            wandb.init(project="nanoGRPO")

        self.metrics = defaultdict(list)

        self.ref_model.to(self.device).to(dtype)

        # self.warmup_compile()
        # print("warmup done!")

    def warmup_compile(self):
        model_inputs = self.tokenizer(["Write a poem about the market crashing in summer"], return_tensors="pt")
        model_inputs = model_inputs.to(self.model.device)
        output_compiled = self.compiled_generate(**model_inputs, generation_config=self.generate_config)
        model_outputs = self.model(**model_inputs)

    def get_per_token_logps(self, model, input_ids) -> Tensor:
        logits = model(input_ids=input_ids).logits
        logits = logits[:, :-1, :]  # Shape: [2, 660, 128256]
        input_ids = input_ids[:, 1:]
        logps = F.log_softmax(logits, dim=-1)
        return torch.gather(logps, -1, input_ids.unsqueeze(-1)).squeeze(-1)

    def compute_loss(self, inputs, old_policy_log_probs, reward, mean_rewards, std_rewards, loss_mask) -> Tensor:
        policy_log_probs = self.get_per_token_logps(self.model, inputs)
        
        # advantage calculation
        advantage: Tensor = (reward - mean_rewards) / (std_rewards + 1e-4)
        advantage = advantage.reshape(-1, 1)

        policy_ratio = torch.exp(policy_log_probs-old_policy_log_probs.detach())

        loss1 = policy_ratio*advantage
        loss2 = torch.clamp(policy_ratio, 1 - self.epsilon, 1 + self.epsilon) * advantage
        loss = -torch.min(loss1, loss2)
        loss = (loss * loss_mask).sum(dim=-1)/ (loss_mask.sum(dim=-1) + 1e-6)
        
        # Only compute KL divergence if beta > 0
        if self.beta > 0:
            with (
                self.ref_model.disable_adapter()
                if self.using_lora  
                else contextlib.nullcontext()
            ):
                ref_policy_log_probs = self.get_per_token_logps(self.ref_model, inputs)
                
            # kl divergence calculation
            log_ratios = ref_policy_log_probs - policy_log_probs
            kld = torch.exp(log_ratios) - log_ratios - 1
            kld = (kld * loss_mask).sum(dim=-1)/ (loss_mask.sum(dim=-1) + 1e-6)
            loss += kld * self.beta
            
            if self.log_wandb:
                for _kd in kld:
                    self.metrics["kld"].append(_kd.mean().item())
                
        return loss.mean(), loss1.mean(), loss2.mean()

    def sample_batch(self):

        inputs_texts = []
        samples = []
        for _ in range(self.batch_size):
            item = next(self.data_loader_iter)
            samples.append(item)
            prompt = item["prompt"]
            formatted = self.tokenizer.apply_chat_template(prompt, tokenize=False, continue_final_message=True)
            inputs_texts.append(formatted)

        encoded = self.tokenizer(inputs_texts, padding=True, return_tensors="pt")
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

        prompt_length = input_ids.shape[1]

        input_ids = torch.repeat_interleave(input_ids, self.group_size, dim=0)
        attention_mask = torch.repeat_interleave(attention_mask, self.group_size, dim=0)
        samples = [sample for _ in range(self.group_size) for sample in samples]

        start_time = time.time()
        outputs = self.model.generate(
            input_ids.to(self.device),
            attention_mask=attention_mask.to(self.device),
            temperature=0.9,
            max_new_tokens=1024,
            top_p=0.9,

            # generation_config=self.generate_config
        )
        end_time = time.time()
        print(f"Time for generation: {end_time - start_time} seconds")


        loss_mask = torch.zeros(outputs.shape, dtype=torch.bool)

        gen_tokens = outputs[:, prompt_length:]
        valid_gen_mask = gen_tokens != self.tokenizer.pad_token_id
        # valid_gen_mask = valid_gen_mask * (gen_tokens != self.tokenizer.eos_token_id)
        loss_mask[:, prompt_length:] = valid_gen_mask

        decoded_outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=False)
        decoded_outputs = [
            d
            .replace(inputs_texts[i // self.group_size], "")
            .replace(self.tokenizer.eos_token, "")
            # .replace(self.tokenizer.bos_token, "")
            .replace(self.tokenizer.pad_token, "")
            for i, d in enumerate(decoded_outputs)
        ]        
        rewards = self.compute_rewards(samples,decoded_outputs)

        avg_decoded_length = valid_gen_mask.sum(dim=-1) / valid_gen_mask.shape[0]
        print(f"Average decoded length: {avg_decoded_length.mean().item()}")
        self.metrics["avg_decoded_length"].append(avg_decoded_length.mean().item())

        return outputs, torch.tensor(rewards, dtype=self.dtype).float(), loss_mask[:, 1:]

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
            idx = self.metrics["idx"][-1]-1
            metrics = {}
            for k, v in self.metrics.items():
                metrics[f"train/{k}"] = v[idx] if len(v) >= idx else v[-1]
                
            wandb.log(metrics)

    def train(self, epochs=1, max_iterations=1000):
        idx = 0
        start_time = time.time()
        while idx < max_iterations:

            x_batch_inputs, rewards, loss_mask = self.sample_batch()
            torch.cuda.empty_cache() 

            


            batch_inputs = x_batch_inputs.reshape(self.batch_size, self.group_size, *x_batch_inputs.shape[1:])
            loss_mask =       loss_mask.reshape(self.batch_size, self.group_size, *loss_mask.shape[1:])
            torch.cuda.empty_cache() # gpu poor hack




            # offload to cpu to save vram
            batch_inputs = batch_inputs.cpu()
            rewards = rewards.cpu()
            loss_mask = loss_mask.cpu()
            torch.cuda.empty_cache() # gpu poor hack

            pi_old = []
            for _, (b_inputs) in enumerate(batch_inputs):
                
                with torch.no_grad():
                    b_old_policy_log_probs = self.get_per_token_logps(self.model, b_inputs.to(self.device)).cpu()
                    torch.cuda.empty_cache()
                    pi_old.append(b_old_policy_log_probs)

            

            for _, (b_inputs, b_old_policy_log_probs, b_reward, b_loss_mask) in enumerate(zip(batch_inputs, pi_old, rewards, loss_mask)):
                idx += 1
                reward = b_reward.to(self.device)
                mean_rewards = reward.mean(dim=-1).unsqueeze(-1)
                std_rewards = reward.std(dim=-1).unsqueeze(-1)
    
                # Remove micro-grouping if group_size == micro_group_size
                if self.group_size == self.micro_group_size:
                    g_inputs = [b_inputs.cpu()]
                    g_old_policy_log_probs = [b_old_policy_log_probs.cpu()]
                    g_reward = [b_reward.cpu()]
                    g_loss_mask = [b_loss_mask.cpu()]
                else:
                    # even groups are too big for VRAM
                    # so we split them into micro groups (it's same as micro batching)
                    g_inputs = b_inputs.reshape(
                        b_inputs.shape[0] // self.micro_group_size,
                        self.micro_group_size,
                        *b_inputs.shape[1:]
                    ).cpu()
                    g_old_policy_log_probs = b_old_policy_log_probs.reshape(
                        b_inputs.shape[0] // self.micro_group_size,
                        self.micro_group_size,
                        *b_old_policy_log_probs.shape[1:]
                    ).cpu()
                    g_reward = b_reward.reshape(
                        b_inputs.shape[0] // self.micro_group_size,
                        self.micro_group_size,
                        *b_reward.shape[1:]
                    ).cpu()
                    g_loss_mask = b_loss_mask.reshape(
                        b_inputs.shape[0] // self.micro_group_size,
                        self.micro_group_size,
                        *b_loss_mask.shape[1:]
                    ).cpu()
                group_losses = []
                group_losses1 = []
                group_losses2 = []
    
                for inputs, old_policy_log_probs, reward, loss_mask in zip(
                    g_inputs, g_old_policy_log_probs, g_reward, g_loss_mask
                ):
                    inputs = inputs.to(self.device)
                    old_policy_log_probs = old_policy_log_probs.to(self.device)
                    reward = reward.to(self.device)
                    loss_mask = loss_mask.to(self.device)
    
                    loss, loss1, loss2 = self.compute_loss(
                        inputs,
                        old_policy_log_probs,
                        reward,
                        mean_rewards,
                        std_rewards,
                        loss_mask
                    )
                    group_losses.append(loss.item())
                    group_losses1.append(loss1.item())
                    group_losses2.append(loss2.item())
                    loss.backward()
                    torch.cuda.empty_cache()    

                self.optimizer.step()
                self.optimizer.zero_grad()

                print(f"{idx:04d} loss: {sum(group_losses)/len(group_losses)} reward: {reward.mean()}")
                if self.log_wandb:
                    self.metrics["idx"].append(idx)
                    self.metrics["total_reward"].append(reward.mean().item())
                    self.metrics["loss"].append(sum(group_losses)/len(group_losses))
                    self.metrics["ratio_loss"].append(loss1.mean().item())
                    self.metrics["clipped_loss"].append(loss2.mean().item())

                torch.cuda.empty_cache()
                
            print(f"iter {idx}  >>> reward: {rewards.mean()}")
            print(f"Total time: {str(datetime.timedelta(seconds=int(time.time() - start_time)))}")
            self.log_metrics()
