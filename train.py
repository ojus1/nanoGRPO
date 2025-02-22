from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model
import torch
from rich import print
import math
from grpo import GRPO


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
# small models are kind of dumb, they need a little push so using this fine-tuned model
# source: https://github.com/joey00072/nanoGRPO/blob/master/cold_start/cold_start_finetune.py
# you can totally use the base model, it will just take longer to converge
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
    return len(s)/1000


dataset = load_dataset("openai/gsm8k", "main")["train"]
dataset = prepare_dataset(dataset)

batch_size = 8
group_size = 8
reward_functions = [
    reward_func_len,
]

print(model)

ref_model = None
trainer = GRPO(
    model,
    ref_model,
    tokenizer=tokenizer,
    group_size=group_size,
    batch_size=batch_size,
    dataset=dataset,
    reward_functions=reward_functions,
)

trainer.train()
