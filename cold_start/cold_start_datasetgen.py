import openai
from openai import OpenAI
import os
import json
from datasets import load_dataset
from tqdm import tqdm


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = "Must answer in following format <thinking>{find answer}</thinking></answer><number>{number}</number>"
dataset = load_dataset("openai/gsm8k", "main")["train"]


def openai_call(prompt):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content


num_generations = 1000
fine_name = "cold_start_datasetgen.json"
lst = []
for data in tqdm(
    dataset, total=min(len(dataset), num_generations), desc="Generating responses"
):
    prompt = data["question"]
    answer = openai_call(prompt)
    lst.append({"question": prompt, "answer": answer})
    if len(lst) > num_generations:
        break

with open(fine_name, "w") as f:
    json.dump(lst, f)
