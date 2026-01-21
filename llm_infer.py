from dotenv import load_dotenv
import re
import os
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import json

load_dotenv(override=True)

verifier_model = os.getenv("VERIFIER_MODEL")

client = OpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
)


# return the reponses and token probabilities
def generate_response(prompt):
    response = client.chat.completions.create(
        model=verifier_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    response_text = response.choices[0].message.content
    return response_text


def llm_check(prompts):
    # parallel generation
    with ThreadPoolExecutor(max_workers=len(prompts)) as executor:
        futures = [executor.submit(generate_response, prompt) for prompt in prompts]
        results = [future.result() for future in futures]
    return results
