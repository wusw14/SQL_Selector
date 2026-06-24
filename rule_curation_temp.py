from memory import Memory
from loader import load_data, load_all_preds
import argparse
import os
import json
import numpy as np
from llm_infer import llm_check
from utils import parse_json
from sentence_transformers import SentenceTransformer


def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="birddev")
    parser.add_argument("--method_name", type=str, default="alphasql")
    parser.add_argument("--rule_mode", type=str, default="relevant")
    parser.add_argument(
        "--retrieval_mode",
        type=str,
        choices=["samedb", "diffdb", "crossds", "all"],
        default="all",
    )
    parser.add_argument("--slice", type=int, default=0)
    return parser.parse_args()


def select_cases(
    question, evidence, text, logic_cond, nl_cond, correct_sqls, incorrect_sqls
):
    correct_sql_str = "\n".join(
        [f"SQL{i+1}: {sql}" for i, sql in enumerate(correct_sqls)]
    )
    incorrect_sql_str = "\n".join(
        [f"SQL{i+len(correct_sqls)+1}: {sql}" for i, sql in enumerate(incorrect_sqls)]
    )
    prompt = f"""You are a helpful assistant that selects the best correct SQL and incorrect SQL to demonstrate the meaning of a specific hint. The selected pair should clearly show how applying the hint distinguishes the correct logic from the incorrect one.

### Context Information
- Hint: {text}
- Logical Condition of Hint: {logic_cond}
- Natural Language Condition of Hint: {nl_cond}
- User's Natural Language Query: {question}
- Evidence/Schema Context: {evidence}

The correct SQLs are: 
{correct_sql_str}

The incorrect SQLs are: 
{incorrect_sql_str}

### Task Instructions
1. Analyze the Hint and its conditions.
2. Select ONE best correct SQL index and ONE best incorrect SQL index. 
3. Criteria for "Best": The chosen incorrect SQL should ideally be a "near-miss" — it fails precisely because it violates the logic specified by the Hint, while the chosen correct SQL perfectly satisfies it.
4. Output your decision in the strict JSON format specified below. Do not include any conversational filler, markdown formatting blocks outside the JSON, or extra text.

#### Output Format:
{{
    "reason": "brief explaination of why these specific SQLs were chosen (under 150 tokens).",
    "pos_sql": "SQL index of the best correct SQL (e.g. 1)",
    "neg_sql": "SQL index of the best incorrect SQL (e.g. 1)"
}}
"""
    response = llm_check([prompt], llm="deepseek")[0]
    print(response)
    result = parse_json(response)
    try:
        pos_sql = correct_sqls[int(result["pos_sql"]) - 1]
        neg_sql = incorrect_sqls[int(result["neg_sql"]) - len(correct_sqls) - 1]
    except:
        return None, None
    return pos_sql, neg_sql


def encode_texts(emb_model, texts):
    embs = emb_model.encode(texts, batch_size=128, device="cuda")
    embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)
    return embs


def if_same(rule1, rule2) -> bool:
    prompt = f"""You are a helpful assistant that determines whether two hints are the same.
Hint 1: 
content: {rule1["text"]}
logic trigger condition: {rule1["logic_cond"]}
natural language trigger condition: {rule1["nl_cond"]}

Hint 2: 
content: {rule2["text"]}
logic trigger condition: {rule2["logic_cond"]}
natural language trigger condition: {rule2["nl_cond"]}

Please determine whether the two hints are the same.
Output Format:
{{
    "reason": "brief explaination of why these two hints are the same or not (under 100 tokens).",
    "same": "Yes/No"
}}
"""
    response = llm_check([prompt], llm="deepseek")[0]
    print(rule1["text"])
    print(rule2["text"])
    print(response)
    result = parse_json(response)
    try:
        return result["same"] == "Yes"
    except:
        return False


if __name__ == "__main__":
    args = parse_option()
    qid_info = load_data(args.dataset_name)
    qid_preds, qid_gp_sql_acc = load_all_preds(args)
    qid_db = {qid: info["db_id"] for qid, info in qid_info.items()}

    folder = f"results/{args.dataset_name}/alphasql/iterative_rules/full_dev"
    input_file = f"{folder}/rule_with_cond_v5.json"
    output_file = f"{folder}/rule_with_cond_v7.json"
    rules = json.load(open(input_file, "r"))

    # rule_with_cases = []
    # for item in rules:
    #     qid = item["source"]
    #     gp_sql_acc = qid_gp_sql_acc[qid]
    #     info = qid_info[qid]
    #     question = info["question"]
    #     evidence = info["evidence"]
    #     gt_sqls = info["SQL"]
    #     db_name = info["db_id"]
    #     correct_sqls, incorrect_sqls = [], []
    #     if type(gt_sqls) == list:
    #         correct_sqls.extend(gt_sqls)
    #     else:
    #         correct_sqls.append(gt_sqls)
    #     for sql_acc in gp_sql_acc:
    #         sqls = sql_acc["sqls"]
    #         acc = sql_acc["acc1"]
    #         if len(sqls) > 2:
    #             sqls = np.random.choice(sqls, 2, replace=False)
    #         if acc == 0:
    #             incorrect_sqls.extend(sqls)
    #     print(
    #         f"correct sqls: {len(correct_sqls)}, incorrect sqls: {len(incorrect_sqls)}"
    #     )
    #     if len(incorrect_sqls) == 0:
    #         continue
    #     # select the case to be the best to demonstrate the usage of the hint
    #     pos_sql, neg_sql = select_cases(
    #         question,
    #         evidence,
    #         item["text"],
    #         item["logic_cond"],
    #         item["nl_cond"],
    #         correct_sqls,
    #         incorrect_sqls,
    #     )
    #     print(f"pos_sql: {pos_sql}, neg_sql: {neg_sql}")
    #     if pos_sql is None or neg_sql is None:
    #         continue
    #     item["pos_sql"] = pos_sql
    #     item["neg_sql"] = neg_sql
    #     rule_with_cases.append(item)
    #     with open(output_file, "w") as f:
    #         json.dump(rule_with_cases, f, indent=2)
    deduplicated_rules = []
    deduplicated_texts = []
    deduplicated_embs = []
    emb_model = SentenceTransformer("BAAI/bge-base-en-v1.5", device="cuda")
    texts = [item["text"] for item in rules]
    embs = encode_texts(emb_model, texts)

    duplicate_cnt = 0
    for i, item in enumerate(rules):
        if len(deduplicated_rules) == 0:
            item["sources"] = [item["source"]]
            item["cases"] = {item["source"]: [item["pos_sql"], item["neg_sql"]]}
            deduplicated_rules.append(item)
            deduplicated_texts.append(item["text"])
            deduplicated_embs.append(embs[i])
            continue
        if item["text"] in deduplicated_texts:
            continue
        # retrieve the top-1 similar rule with embeddings and check whether they are the same
        sim_score = np.dot(embs[i], np.array(deduplicated_embs).T)
        max_sim_index = np.argmax(sim_score)
        max_sim_rule = deduplicated_rules[max_sim_index]
        max_sim_score = sim_score[max_sim_index]
        if max_sim_score < 0.8:
            item["sources"] = [item["source"]]
            item["cases"] = {item["source"]: [item["pos_sql"], item["neg_sql"]]}
            deduplicated_rules.append(item)
            deduplicated_texts.append(item["text"])
            deduplicated_embs.append(embs[i])
            continue
        # check by the LLM
        print(f"Hint {i+1} is checking")
        if if_same(max_sim_rule, item) == False:  # not the same
            item["sources"] = [item["source"]]
            item["cases"] = {item["source"]: [item["pos_sql"], item["neg_sql"]]}
            deduplicated_rules.append(item)
            deduplicated_texts.append(item["text"])
            deduplicated_embs.append(embs[i])
        else:
            # update the sources
            deduplicated_rules[max_sim_index]["sources"].append(item["source"])
            # update the logic condition of the max_sim_rule
            org_logic_cond = max_sim_rule["logic_cond"]
            cur_logic_cond = item["logic_cond"]
            new_logic_cond = f"{org_logic_cond} | {cur_logic_cond}"
            deduplicated_rules[max_sim_index]["logic_cond"] = new_logic_cond
            deduplicated_rules[max_sim_index]["cases"][item["source"]] = [
                item["pos_sql"],
                item["neg_sql"],
            ]
            duplicate_cnt += 1
        with open(output_file, "w") as f:
            json.dump(deduplicated_rules, f, indent=2)
    print(f"duplicate cnt: {duplicate_cnt}")
