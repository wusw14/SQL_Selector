from loader import load_data, load_preds
import sys
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import json
import os
import time
from dotenv import load_dotenv
from parser import SQLCollection
from memory import Memory
from database import Database

load_dotenv(override=True)

verifier_model = os.getenv("VERIFIER_MODEL")

client = OpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
)
N = 3


def format_prompt(question, evidence, schema, pred_sql):
    # COT prompt
    prompt = f"""You are a SQL and natural language semantic analysis expert. Given a natural language query (NL query), related supplementary information (evidence), and a generated SQL query, analyze whether the generated SQL query is correct.

Schema:
{schema}

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


# Return the responses and token probabilities
def generate_response(prompt):
    response = client.chat.completions.create(
        model=verifier_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8,
        logprobs=True,  # Enable logprobs to get token probabilities
        top_logprobs=2,  # Get top 2 tokens' probabilities
        extra_body={
            "top_k": 20,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )

    response_text = response.choices[0].message.content

    # Get logprobs for all generated tokens
    logprobs_content = response.choices[0].logprobs.content

    # Get probability for the Yes/No token
    prob_normalized = get_normalized_probability(logprobs_content)

    return prob_normalized


# If you want normalized probability (relative probability between Yes and No):
def get_normalized_probability(logprobs_content):
    """
    Get normalized Yes vs No probability: P(Yes) / (P(Yes) + P(No))
    """
    yes_logprob = None
    no_logprob = None

    # Find tokens containing "Yes" or "No"
    for token_logprob in reversed(logprobs_content):
        token_lower = token_logprob.token.lower().strip("\"' :,\n")

        if (yes_logprob is None and "yes" in token_lower) or (
            no_logprob is None and "no" in token_lower
        ):
            top_logprobs = token_logprob.top_logprobs
            for logprob in top_logprobs:
                if "yes" in logprob.token.lower().strip("\"' :,\n"):
                    yes_logprob = logprob.logprob
                elif "no" in logprob.token.lower().strip("\"' :,\n"):
                    no_logprob = logprob.logprob
            break

    # Handle cases where only one token is found
    if yes_logprob is None and no_logprob is None:
        return 0.5  # Neither found, return neutral probability
    elif yes_logprob is None:
        return 0.0  # Only "No" found, "Yes" probability near zero
    elif no_logprob is None:
        return 1.0  # Only "Yes" found, "No" probability near zero

    # Calculate normalized probability
    yes_prob = np.exp(yes_logprob)
    no_prob = np.exp(no_logprob)
    return yes_prob / (yes_prob + no_prob)


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
    output_file = os.path.join(output_dir, f"{method_name}_{model_name}_GenRM-CoT.json")
    eval_dir = f"eval_results/{dataset_name}/{method_name}/{model_name}"
    qid_sql_acc_file = (
        f"eval_results/{dataset_name}/{method_name}/{model_name}/gp_sql_acc.json"
    )
    qid_sql_acc = json.load(open(qid_sql_acc_file, "r"))

    if os.path.exists(output_file):
        qid_sqls_probs = json.load(open(output_file, "r"))
    else:
        qid_sqls_probs = {}

    eval_base = json.load(open(os.path.join(eval_dir, "majority.json"), "r"))
    qid_to_be_checked = []
    for qid, res in eval_base.items():
        if res["upper_acc"] == 0 or res["lower_acc"] == 1 or qid in qid_sqls_probs:
            continue
        qid_to_be_checked.append(int(qid))
    print(f"len(qid_to_be_checked): {len(qid_to_be_checked)}")

    db_memory = Memory()

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
        db_name = info["db_id"]

        if db_name not in db_memory.memory:
            db = Database(dataset_name, db_name)
            db_memory.add(db_name, db)
            print("=====joinable columns=====")
            joinable_column_sets = set()
            for key, values in db.joinable_columns.items():
                joinable_column_sets.add(frozenset(values))
            for column_set in joinable_column_sets:
                print(f"{column_set}")
            # print(db.display())
            print("\n" * 5)
        else:
            db = db_memory.get(db_name)

        sql_collection = SQLCollection(preds, db, info)
        schema = sql_collection.schema_note

        if len(preds) > 0:
            prompts = []
            for sql in preds:
                for _ in range(N):
                    prompts.append(format_prompt(question, evidence, schema, sql))
            probs = llm_check(prompts)
            probs = np.array(probs).reshape(len(preds), N)
            probs_raw = np.round(probs, 5).tolist()
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
            "probs_raw": probs_raw,
            "time_cost": time.time() - start_time,
        }

        with open(output_file, "w") as f:
            json.dump(qid_sqls_probs, f, indent=2)

    with open(output_file, "w") as f:
        json.dump(qid_sqls_probs, f, indent=2)
