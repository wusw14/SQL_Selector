from dotenv import load_dotenv
import re
import os
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import json

load_dotenv(override=True)

verifier_model = os.getenv("VERIFIER_MODEL")
rule_gen_model = os.getenv("RULE_GEN_MODEL")

base_client = OpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
)
rule_gen_client = OpenAI(
    api_key=os.getenv("RULE_GEN_API_KEY"),
    base_url=os.getenv("RULE_GEN_URL"),
)


# return the reponses and token probabilities
def generate_response(prompt, llm="Qwen3-30B"):
    if llm == "deepseek":
        client = rule_gen_client
        model = rule_gen_model
        temperature = 1.0
    else:
        client = base_client
        model = verifier_model
        temperature = 0.7
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        extra_body={
            "top_k": 20,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    return response.choices[0].message.content


def llm_check(prompts, llm="Qwen3-30B"):
    # parallel generation
    with ThreadPoolExecutor(max_workers=len(prompts)) as executor:
        futures = [
            executor.submit(generate_response, prompt, llm) for prompt in prompts
        ]
        results = [future.result() for future in futures]
    return results
