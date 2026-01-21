from loader import load_data, load_preds
import sys
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import json
import os
import time
from dotenv import load_dotenv
import re

load_dotenv(override=True)

verifier_model = os.getenv("VERIFIER_MODEL")

client = OpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
)
N = 1


def format_prompt(question, evidence, pred_sql):
    # COT prompt
    prompt = f"""You are a SQL and natural language semantic analysis expert. Given a natural language query (NL query), related supplementary information (evidence), and a generated SQL query, analyze whether the generated SQL query is correct.
Query: {question}
Evidence: {evidence}
Generated SQL: {pred_sql}

Please check if the generated SQL query is correct for the given query. You might refer to the evidence if available.
Your response should be in the json format:
{{
    "think": "Briefly explain your reasoning process.",
    "correctness": "Yes/No"
}}
    """
    return prompt


# return the reponses and token probabilities
def generate_response(prompt):
    response = client.chat.completions.create(
        model=verifier_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8,
    )
    response_text = response.choices[0].message.content
    # TODO: parse the correctness from the response
    try:
        start_idx = response_text.find("{")
        end_idx = response_text.rfind("}") + 1
        response_text = response_text[start_idx:end_idx]
        response_text = json.loads(response_text)
        correctness = response_text["correctness"]
        think = response_text["think"]
    except:
        print(response_text)
        correctness = "No"
        think = ""
    if correctness == "Yes":
        prob = 1
    else:
        prob = 0
    return {"correctness": correctness, "prob": prob, "think": think}


def llm_check(prompts):
    # parallel generation
    with ThreadPoolExecutor(max_workers=len(prompts)) as executor:
        futures = [executor.submit(generate_response, prompt) for prompt in prompts]
        results = [future.result() for future in futures]
    correctness = [result["correctness"] for result in results]
    probs = [result["prob"] for result in results]
    thinks = [result["think"] for result in results]
    return correctness, probs, thinks


if __name__ == "__main__":
    method_name = sys.argv[1]
    dataset_name = sys.argv[2]
    model_name = sys.argv[3]
    qid_info = load_data(dataset_name)
    qid_preds, _ = load_preds(method_name, dataset_name, model_name)
    print(len(qid_preds))
    verifier = os.getenv("MODEL_ABBR")
    output_dir = f"results/{verifier}/{dataset_name}/cot"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{method_name}_{model_name}_probs.json")

    if os.path.exists(output_file):
        qid_sqls_probs = json.load(open(output_file, "r"))
    else:
        qid_sqls_probs = {}
    for qid, preds in qid_preds.items():
        if qid in qid_sqls_probs or str(qid) in qid_sqls_probs:
            continue
        start_time = time.time()
        info = qid_info[qid]
        question = info["question"]
        evidence = info["evidence"]
        gt_sql = info["SQL"]

        if len(preds) > 0:
            prompts = []
            for sql in preds:
                for _ in range(N):
                    prompts.append(format_prompt(question, evidence, sql))
            correctness, probs, thinks = llm_check(prompts)
            probs = np.array(probs).reshape(len(preds), N)
            probs = np.round(np.mean(probs, axis=1), 5).tolist()
            if len(qid_sqls_probs) == 0:
                print(prompts[0])
        else:
            probs = []
        qid_sqls_probs[qid] = {
            "gt_sql": gt_sql,
            "sqls": preds,
            "correctness": correctness,
            "thinks": thinks,
            "probs": probs,
            "time_cost": time.time() - start_time,
        }

        with open(output_file, "w") as f:
            json.dump(qid_sqls_probs, f, indent=2)
