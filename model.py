from typing import assert_type
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model
import torch
import time
from rich import print
from torch import Tensor
import contextlib
import torch.nn.functional as F

class GRPO:
    def __init__(
        self,
        model,
        ref_model,
        group_size=8,
        batch_size=1,
        dataset=None,
        reward_functions=None,
        dtype=torch.bfloat16,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.dataset = dataset.shuffle(seed=42)
        self.data_loader_iter = iter(self.dataset)
        self.group_size = group_size
        self.batch_size = batch_size
        self.dtype = dtype
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-5)
        print(
            f"Total parameters: {sum(p.numel() for p in self.model.parameters()) / 1e6} M parameters"
        )

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
        print(f"{logits.shape=} {input_ids.shape=}")

        logits = logits[:, :-1, :]  # Shape: [2, 660, 128256]
        input_ids = input_ids[:, 1:] 
        logps = F.log_softmax(logits, dim=-1)
        return torch.gather(logps, -1, input_ids.unsqueeze(-1)).squeeze(-1)

    
    def compute_loss(self) -> Tensor:
        batch_inputs: Tensor
        rewards: list[int]
        batch_inputs, rewards, loss_mask = self.sample_batch()
        print(f"{rewards.shape=}")
        batch_inputs = batch_inputs.reshape(
             self.batch_size, self.group_size, *batch_inputs.shape[1:]
        )

        for _, (inputs, reward) in enumerate(zip(batch_inputs, rewards)):
            
            policy_log_probs = self.get_per_token_logps(self.model, inputs)
            with (
                self.ref_model.disable_adapter()
                if self.using_lora
                else contextlib.nullcontext()
            ):
                ref_policy_log_probs = self.get_per_token_logps(self.ref_model, inputs)

            log_ratios =  ref_policy_log_probs - policy_log_probs

            kld = torch.exp(log_ratios) - log_ratios - 1

            mean_rewards = reward.mean(dim=-1).unsqueeze(-1)
            std_rewards = reward.std(dim=-1).unsqueeze(-1)
            advantage: Tensor = (mean_rewards - reward) / std_rewards
            advantage = advantage.reshape(-1,1)
            
            print(f"{advantage.shape=} {policy_log_probs.shape=} {reward.shape=} {policy_log_probs.shape=}")
            loss: Tensor = (advantage * policy_log_probs).sum() 
            loss += kld.mean()
            loss.backward()

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
        outputs = self.model.generate(
            input_ids.to(self.device),
            min_new_tokens=512,
            max_new_tokens=512,
            temperature=1.1,
        )
        end_time = time.time()
        print(f"Time taken: {end_time - start_time} seconds")

        decoded_outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
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

    def train(self, epochs=1):
        for epoch in range(epochs):
            for batch in self.data_loader_iter:
                self.optimizer.zero_grad()
                loss = self.compute_loss()
                print(f"Epoch {epoch + 1} loss: {loss}")
                self.optimizer.step()


SYSTEM_PROMPT = "Must answer in following format <thinking>...</thinking><answer>...</answer><number>...</number>"


def prepare_dataset(dataset) -> Dataset:
    extract_hash_answer = (
        lambda text: text.split("####")[1].strip() if "####" in text else None
    )

    def process_example(example: dict):
        answer = extract_hash_answer(example["answer"])
        if answer is None:
            return None
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["question"]},
            ],
            "answer": answer,
        }

    dataset = dataset.map(
        process_example,
        remove_columns=[
            col for col in dataset.column_names if col not in ["prompt", "answer"]
        ],
    )
    dataset = dataset.filter(lambda x: x is not None)

    return dataset


# model_name = "Qwen/Qwen2.5-0.5B-Instruct"
# small models are kind of dumb, they need littel push so using this filetuned model
# source: https://github.com/joey00072/nanoGRPO/blob/main/cold_start/cold_start_finetune.py
model_name = "joey00072/Llama-3.2-1B-Instruct-cold-start-ft"

model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

lora_config = LoraConfig(
    r=2,
    lora_alpha=16,
    lora_dropout=0.1,
    use_dora=True,
)
model = get_peft_model(model, lora_config)
model = model.to(torch.bfloat16)


def reward_func_len(s: str):
    return len(s)


dataset = load_dataset("openai/gsm8k", "main")["train"]
dataset = prepare_dataset(dataset)

batch_size = 2
group_size = 3
reward_functions = [
    reward_func_len,
]

print(model)
trainer = GRPO(model, None,  group_size,batch_size, dataset, reward_functions)

trainer.train()
