from loader import load_data, load_preds
import sys
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import json
import os
import time
from dotenv import load_dotenv

load_dotenv(override=True)

verifier_model = os.getenv("VERIFIER_MODEL")

client = OpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
)
N = 3


def format_prompt(question, evidence, pred_sql):
    prompt = f"""You are a SQL and natural language semantic analysis expert. Given a natural language query (NL query), related supplementary information (evidence), and a generated SQL query, analyze whether the generated SQL query is correct.
Query: {question}
Evidence: {evidence}
Generated SQL: {pred_sql}
Is the generated SQL query correct? Directly answer "Yes" or "No" without any other explanation.
    """
    return prompt


# return the reponses and token probabilities
def generate_response(prompt):
    response = client.chat.completions.create(
        model=verifier_model,
        messages=[{"role": "user", "content": prompt}],
        stream=False,
        temperature=0.8,
        logprobs=True,
        top_logprobs=2,
        extra_body={
            "top_k": 20,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    response_text = response.choices[0].message.content
    # print("[DEBUG] logprobs:", response.choices[0].logprobs)
    response_logprobs = response.choices[0].logprobs.content[0].top_logprobs
    yes, no = None, None
    for logprob in response_logprobs:
        if yes is None and (logprob.token == "Yes" or logprob.token == "yes"):
            yes = logprob.logprob
        if no is None and (logprob.token == "No" or logprob.token == "no"):
            no = logprob.logprob
    if yes is None:
        prob = 0
    elif no is None:
        prob = 1
    else:
        prob = np.exp(yes) / (np.exp(yes) + np.exp(no))
    return prob


def llm_check(prompts):
    # parallel generation
    with ThreadPoolExecutor(max_workers=len(prompts)) as executor:
        futures = [executor.submit(generate_response, prompt) for prompt in prompts]
        probs = [future.result() for future in futures]
    return probs


if __name__ == "__main__":
    method_name = sys.argv[1]
    dataset_name = sys.argv[2]
    model_name = sys.argv[3]
    qid_info = load_data(dataset_name)
    qid_preds, _ = load_preds(method_name, dataset_name, model_name)
    print(len(qid_preds))
    verifier = os.getenv("MODEL_ABBR")
    output_dir = f"results/{verifier}/{dataset_name}"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{method_name}_{model_name}_probs.json")
    eval_dir = f"eval_results/{dataset_name}/{method_name}/{model_name}"
    qid_sql_acc_file = (
        f"eval_results/{dataset_name}/{method_name}/{model_name}/gp_sql_acc.json"
    )
    qid_sql_acc = json.load(open(qid_sql_acc_file, "r"))

    eval_base = json.load(open(os.path.join(eval_dir, "exec.json"), "r"))
    qid_to_be_checked = []
    for qid, res in eval_base.items():
        if res["upper_acc"] == 0 or res["lower_acc"] == 1:
            continue
        qid_to_be_checked.append(int(qid))
    print(f"len(qid_to_be_checked): {len(qid_to_be_checked)}")

    if os.path.exists(output_file):
        qid_sqls_probs = json.load(open(output_file, "r"))
    else:
        qid_sqls_probs = {}
    for qid, preds in qid_preds.items():
        if qid in qid_sqls_probs or str(qid) in qid_sqls_probs:
            continue
        if qid not in qid_to_be_checked:
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
            probs = llm_check(prompts)
            probs = np.array(probs).reshape(len(preds), N)
            probs = np.round(np.mean(probs, axis=1), 5).tolist()
            # if len(qid_sqls_probs) == 0:
            #     print(prompts[0])
        else:
            probs = []
        if len(probs) == 0:
            selected_sql = "None"
            selected_acc = 0
        else:
            selected_sql = preds[np.argmax(probs)]
            gp_acc_list = qid_sql_acc[str(qid)]
            sql_acc_dict = {}
            for sql_acc in gp_acc_list:
                sqls = sql_acc["sqls"]
                acc = sql_acc["acc1"]
                for sql in sqls:
                    sql_acc_dict[sql] = acc
            selected_acc = sql_acc_dict.get(selected_sql, 0)

        qid_sqls_probs[qid] = {
            "gt_sql": gt_sql,
            "selected_sql": selected_sql,
            "selected_acc": selected_acc,
            "sqls": preds,
            "probs": probs,
            "time_cost": time.time() - start_time,
        }

        with open(output_file, "w") as f:
            json.dump(qid_sqls_probs, f, indent=2)
