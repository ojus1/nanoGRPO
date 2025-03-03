from transformers import AutoTokenizer
from qwen2_latent import Qwen2ForCausalLM
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model
import torch
from rich import print
from coconut_grpo import GRPO
import re

SYSTEM_PROMPT = '''You think very carefully in your latent space before speaking out loud.
You will have <num_thoughts=N> computation budget. 
Think about the problem at length and use Breadth-First Search approach to solving any problem.
After your latent thoughts, you will respond with <answer>NUMBER</answer> format.
Your answer must provide a single number inside the <answer> tags. Don't continue your response after the </answer> tag.'''
 
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


model_name = "Qwen/Qwen2.5-3B-Instruct"
# small models are kind of dumb, they need a little push so using this fine-tuned model
# source: https://github.com/joey00072/nanoGRPO/blob/master/cold_start/cold_start_finetune.py
# you can totally use the base model, it will just take longer to converge
# model_name = "joey00072/Llama-3.2-1B-Instruct-cold-start-ft2"

# model_name = "unsloth/Llama-3.2-3B-Instruct"


model = Qwen2ForCausalLM.from_pretrained(model_name, attn_implementation="sdpa")
tokenizer = AutoTokenizer.from_pretrained(model_name)
# Add special tokens for latent reasoning
special_tokens = {"additional_special_tokens": ["<bot>", "<eot>"]}
tokenizer.add_special_tokens(special_tokens)
model.bot_token_id = tokenizer.convert_tokens_to_ids("<bot>")
model.eot_token_id = tokenizer.convert_tokens_to_ids("<eot>")

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules='all-linear',
    # target_modules=['q_proj', 'v_proj'],
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
    try:
        answer = sample["answer"]
        extracted_response = extract_xml_answer(response)
        answer = answer.strip().replace(" ", "").replace(",", "")
        # print("Inside correctness_reward_func; answer: ", answer)
        # print("Inside correctness_reward_func; response: ", extracted_response)
        return 2.0 if float(extracted_response) == float(answer) else -1.0
    except Exception as e:
        print("Error in correctness_reward_func: ", e)
        return -1.0

import random

def format_reward_func(sample, response):
    if random.random() < 0.3:
        # log 30% of the time
        print("Logging format reward")
        print("answer: ", sample["answer"])
        print("response: ", response)
    format_reward = -1
    if response.count("<answer>") == 1 and response.count("</answer>") == 1 and response.find("<answer>") < response.find("</answer>"):
        # Check if there's text after </answer>
        end_tag_pos = response.find("</answer>") + len("</answer>")
        if end_tag_pos < len(response.strip()):
            format_reward = -0.75
        else:
            format_reward = 1.0
    elif response.count("<answer>") == 1 or response.count("</answer>") == 1:
        format_reward = -0.5
    return format_reward
dataset = load_dataset("openai/gsm8k", "main")["train"]
dataset = prepare_dataset(dataset)

group_size = 8
# micro_group_size = 8
latent_warmup_ratio = 0.001
batch_size = 8
lr = 2e-5
weight_decay = 0.1
reward_functions = [
    correctness_reward_func,
    format_reward_func,
]

print(model)

ref_model = None
trainer = GRPO(
    model,
    ref_model,
    tokenizer=tokenizer,
    group_size=group_size,
    latent_warmup_ratio=latent_warmup_ratio,
    max_iterations=5000,
    dataset=dataset,
    reward_functions=reward_functions,
    log_wandb=True,
    lr=lr,
    weight_decay=weight_decay,
    batch_size=batch_size,
    max_continuous_tokens=16,
    noise_scale=0.1,
    forced_prefix="<answer>",
    stop_token="</answer>",
)

trainer.train()
