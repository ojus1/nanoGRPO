import torch
import torch.nn.functional as F
from torch import Tensor
import contextlib
import time


class GRPO:
    def __init__(
        self,
        model,
        ref_model,
        tokenizer=None,
        group_size=8,
        batch_size=1,
        dataset=None,
        reward_functions=None,
        dtype=None,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.dataset = dataset.shuffle(seed=42)
        self.data_loader_iter = iter(self.dataset)
        self.group_size = group_size
        self.micro_group_size = 2
        self.batch_size = batch_size
        self.dtype = dtype if dtype is not None else (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
        self.beta = 1e-2
        self.epsilon = 0.1
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=3e-4)
        assert reward_functions is not None, "Must pass reward_functions"
        self.reward_functions: list = reward_functions

        self.using_lora = True if self.ref_model is None else False
        if self.using_lora:
            self.ref_model = model

        self.distributed = False

        self.model.to(self.device).to(dtype)
        self.ref_model.to(self.device).to(dtype)

    def get_per_token_logps(self, model, input_ids) -> Tensor:
        logits = model(input_ids=input_ids).logits
        logits = logits[:, :-1, :]  # Shape: [2, 660, 128256]
        input_ids = input_ids[:, 1:]
        logps = F.log_softmax(logits, dim=-1)
        return torch.gather(logps, -1, input_ids.unsqueeze(-1)).squeeze(-1)

    def compute_loss(self, inputs, old_policy_log_probs, reward, loss_mask) -> Tensor:
        mean_rewards = reward.mean(dim=-1).unsqueeze(-1)
        std_rewards = reward.std(dim=-1).unsqueeze(-1)

        g_inputs = inputs.reshape(inputs.shape[0]//self.micro_group_size,self.micro_group_size, *inputs.shape[1:]).cpu()
        old_policy_log_probs = old_policy_log_probs.reshape(inputs.shape[0]//self.micro_group_size,self.micro_group_size, *old_policy_log_probs.shape[1:]).cpu()
        reward = reward.reshape(inputs.shape[0]//self.micro_group_size,self.micro_group_size, *reward.shape[1:]).cpu()
        # loss_mask = loss_mask.reshape(self.micro_group_size,inputs.shape[0]//self.micro_group_size, *loss_mask.shape[1:])
        torch.cuda.empty_cache()

        print(f"{inputs.shape=} {old_policy_log_probs.shape=} {reward.shape=} {loss_mask.shape=}")
        for inputs, old_policy_log_probs, reward, loss_mask in zip(g_inputs, old_policy_log_probs, reward, loss_mask):
            print(f"{inputs.shape=} {old_policy_log_probs.shape=} {reward.shape=} {loss_mask.shape=}")
            torch.cuda.empty_cache()
            inputs = inputs.to(self.device)
            old_policy_log_probs = old_policy_log_probs.to(self.device)
            reward = reward.to(self.device)
            # loss_mask = loss_mask.to(self.device)
            policy_log_probs = self.get_per_token_logps(self.model, inputs)
            with (
                self.ref_model.disable_adapter()
                if self.using_lora  
                else contextlib.nullcontext()
            ):
                ref_policy_log_probs = self.get_per_token_logps(self.ref_model, inputs)


            # advantage calculation
            advantage: Tensor = (reward - mean_rewards) / (std_rewards + 1e-6)
            advantage = advantage.reshape(-1, 1)

            # policy ratio calculation
            policy_ratio = torch.exp(policy_log_probs - old_policy_log_probs)
            loss1 = policy_ratio * advantage
            loss2 = torch.clamp(policy_ratio, 1 - self.epsilon, 1 + self.epsilon) * advantage
            loss = torch.min(loss1, loss2)

            # kl divergence calculation
            log_ratios = ref_policy_log_probs - policy_log_probs
            kld = torch.exp(log_ratios) - log_ratios - 1

            loss = loss 
            loss += kld * self.beta
            loss = loss.mean()
            loss.backward()
            torch.cuda.empty_cache()

        return loss

    def sample_batch(self):
        if self.distributed:
            return self.distributed_sample_batch()

        inputs_texts = []
        for _ in range(self.batch_size):
            prompt = next(self.data_loader_iter)["prompt"]
            formatted = self.tokenizer.apply_chat_template(prompt, tokenize=False)
            inputs_texts.append(formatted)

        encoded = self.tokenizer(inputs_texts, padding=True, return_tensors="pt")
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

        prompt_length = input_ids.shape[1]

        input_ids = torch.repeat_interleave(input_ids, self.group_size, dim=0)

        start_time = time.time()
        max_new_tokens = 512   
        outputs = self.model.generate(
            input_ids.to(self.device),
            min_new_tokens=512,
            max_new_tokens=max_new_tokens,
            temperature=1.1,
        )
        end_time = time.time()
        print(f"Time taken: {end_time - start_time} seconds")

        decoded_outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        print(decoded_outputs[0])
        rewards = self.compute_rewards(decoded_outputs)

        loss_mask = torch.zeros(outputs.shape, dtype=torch.bool)

        gen_tokens = outputs[:, prompt_length:]
        valid_gen_mask = gen_tokens != self.tokenizer.pad_token_id
        loss_mask[:, prompt_length:] = valid_gen_mask

        print(f"{loss_mask.shape=} {outputs.shape=} {prompt_length=}")
        return outputs, torch.tensor(rewards, dtype=self.dtype).float(), loss_mask

    def compute_rewards(self, samples) -> list:
        rewards = [[] for _ in range(self.batch_size)]
        for idx, sample in enumerate(samples):
            reward = 0
            for func in self.reward_functions:
                reward += func(sample)
            rewards[idx % self.batch_size].append(reward)
        return torch.tensor(rewards, dtype=self.dtype).to(self.device)

    def train(self, epochs=1, max_iterations=1000):
        idx = 0
        while idx < max_iterations:
            batch_inputs, rewards, loss_mask = self.sample_batch()
            batch_inputs = batch_inputs.reshape(self.batch_size, self.group_size, *batch_inputs.shape[1:])
            
            torch.cuda.empty_cache() # gpu poor hack
            pi_old = []
            with torch.no_grad():
                for inputs in batch_inputs:
                    pi_old.append(self.get_per_token_logps(self.model, inputs).cpu())
                    torch.cuda.empty_cache() 

            pi_old = torch.stack(pi_old)
            # offload to cpu to save vram
            batch_inputs = batch_inputs.cpu() 
            rewards = rewards.cpu()
            loss_mask = loss_mask.cpu()
            torch.cuda.empty_cache() # gpu poor hack

            for _, (inputs, old_policy_log_probs, reward) in enumerate(zip(batch_inputs, pi_old, rewards)):
                idx += 1
                
                inputs = inputs.to(self.device)
                old_policy_log_probs = old_policy_log_probs.to(self.device)
                reward = reward.to(self.device)
                
                loss = self.compute_loss(inputs,old_policy_log_probs, reward, loss_mask)
                # loss.backward()

                self.optimizer.step()
                self.optimizer.zero_grad()

                print(f"{idx:04d} loss: {loss} reward: {reward.mean()}")
                torch.cuda.empty_cache()

            print(f"Epoch {idx}  >>> reward: {reward.mean()}")
