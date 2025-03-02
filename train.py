from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model
import torch
from rich import print
from grpo import GRPO
import re

SYSTEM_PROMPT = "Respond in following format. Answer should only be a number (no units!):<thinking>{step by step reasoning}</thinking><answer>{number}</answer>"


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
                {"role": "assistant", "content": "<thinking>"},
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


model_name = "Qwen/Qwen2.5-1.5B-Instruct"
# small models are kind of dumb, they need a little push so using this fine-tuned model
# source: https://github.com/joey00072/nanoGRPO/blob/master/cold_start/cold_start_finetune.py
# you can totally use the base model, it will just take longer to converge
# model_name = "joey00072/Llama-3.2-1B-Instruct-cold-start-ft2"

# model_name = "unsloth/Llama-3.2-3B-Instruct"


model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="sdpa")
tokenizer = AutoTokenizer.from_pretrained(model_name)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules='all-linear',
    # use_dora=True,
    # lora_dropout=0.1,
)
model = get_peft_model(model, lora_config)
model = model.to(torch.bfloat16)


def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

# Reward functions
def correctness_reward_func(sample, response):
    answer = sample["answer"]
    extracted_response = extract_xml_answer(response)
    return 2.0 if extracted_response == answer else 0.0

def int_reward_func(sample, response):
    extracted_response = extract_xml_answer(response)
    return 0.5 if extracted_response.isdigit() else 0.0

def strict_format_reward_func(sample, response):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<thinking>\n.*?\n</thinking>\n<answer>\n.*?\n</answer>\n$"
    match = re.match(pattern, response)
    return 0.5 if match else 0.0

def soft_format_reward_func(sample, response):
    """Reward function that checks if the completion has a specific format."""
    print("-"*100)
    print(response)
    print("-"*100)
    pattern = r"<thinking>.*?</thinking>\s*<answer>.*?</answer>"
    match = re.search(pattern, response, re.DOTALL)
    return 0.5 if match else 0.0

def xmlcount_reward_func(sample, response):
    count = 0.0
    if response.count("<thinking>\n") == 1:
        count += 0.125
    if response.count("\n</thinking>\n") == 1:
        count += 0.125
    if response.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(response.split("\n</answer>\n")[-1])*0.001
    if response.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(response.split("\n</answer>")[-1]) - 1)*0.001
    return count


dataset = load_dataset("openai/gsm8k", "main")["train"]
dataset = prepare_dataset(dataset)

group_size = 8
micro_group_size = 4
batch_size = 4
lr = 2e-5
weight_decay = 0.1
reward_functions = [
    xmlcount_reward_func,
    soft_format_reward_func,
    strict_format_reward_func,
    int_reward_func,
    correctness_reward_func,
]

print(model)

ref_model = None
trainer = GRPO(
    model,
    ref_model,
    tokenizer=tokenizer,
    group_size=group_size,
    micro_group_size=micro_group_size,
    dataset=dataset,
    reward_functions=reward_functions,
    log_wandb=True,
    lr=lr,
    weight_decay=weight_decay,
    batch_size=batch_size,
)

trainer.train()
