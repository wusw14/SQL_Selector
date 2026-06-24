from typing import List, Dict
import numpy as np
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
from prompts import get_rule_logic_cond
from llm_infer import llm_check
from utils import parse_json
import json
import argparse
from loader import load_all_preds, load_data
from collections import defaultdict
import os
from rule_lib import Rule, RuleCollection


def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="birddev")
    parser.add_argument("--method_name", type=str, default="alphasql")
    parser.add_argument("--rule_mode", type=str, default="relevant")
    return parser.parse_args()


def gen_logic_cond(rule: Rule, query: str, evidence: str, sqls: List[str]) -> Rule:
    rule_cond_prompt = get_rule_logic_cond(rule.text, query, evidence, sqls)
    # print(f"{rule_cond_prompt}\n\n")
    rule_cond_response = llm_check([rule_cond_prompt], llm="deepseek")[0]
    rule_cond_response = parse_json(rule_cond_response)
    print(rule_cond_response)
    try:
        rule.logic_cond = rule_cond_response["logic_cond"]
        rule.nl_cond = rule_cond_response["nl_cond"]
    except:
        rule.logic_cond = None
        rule.nl_cond = None
    return rule


if __name__ == "__main__":
    database = "full_dev"
    args = parse_option()
    rule_file = f"results/{args.dataset_name}/alphasql/iterative_rules/{database}/low_quality_score_v2.json"
    qid_preds, qid_gp_sql_acc = load_all_preds(args)
    qid_info = load_data(args.dataset_name)
    input_rules = json.load(open(rule_file, "r"))
    rules = []
    for item in input_rules:
        # if item["score_diff"] <= 0:
        #     continue
        rule_obj = Rule(item["rule"], [item["source"]], None, None, 0.5)
        rules.append(rule_obj)
    output_file = f"results/{args.dataset_name}/alphasql/iterative_rules/{database}/low_quality_rule_with_cond_v6.json"
    if os.path.exists(output_file):
        rule_with_cond = json.load(open(output_file, "r"))
        processed_rules = [item["text"] for item in rule_with_cond]
    else:
        rule_with_cond = []
        processed_rules = []
    print(f"processed_rules: {len(processed_rules)}/{len(rules)}")
    for rule in rules:
        if rule.text in processed_rules:
            continue
        qid = rule.sources[0]
        gp_sql_acc = qid_gp_sql_acc[qid]
        info = qid_info[qid]
        question = info["question"]
        evidence = info["evidence"]
        gt_sqls = info["SQL"]
        # group sql into correct and incorrect
        correct_sqls, incorrect_sqls = [], []
        if type(gt_sqls) == list:
            correct_sqls.extend(gt_sqls)
        else:
            correct_sqls.append(gt_sqls)
        for sql_acc in gp_sql_acc:
            sqls = sql_acc["sqls"]
            acc = sql_acc["acc1"]
            if len(sqls) > 2:
                sqls = np.random.choice(sqls, 2, replace=False)
            if acc == 1:
                correct_sqls.extend(sqls)
            else:
                incorrect_sqls.extend(sqls)
        print(
            f"correct sqls: {len(correct_sqls)}, incorrect sqls: {len(incorrect_sqls)}"
        )
        if len(incorrect_sqls) == 0:
            continue
        preds = correct_sqls[:3] + incorrect_sqls[:3]
        if len(correct_sqls) == 0 or len(incorrect_sqls) == 0:
            continue
        print(len(correct_sqls))
        print(len(incorrect_sqls))
        print(f"Hint: {rule.text}")
        print(f"NL query: {question}")
        print(f"Evidence: {evidence}")
        print(f"SQLs: {preds}")
        print(f"--------\n\n")
        rule_obj = gen_logic_cond(rule, question, evidence, preds)
        print(f"logic_cond: {rule.logic_cond}")
        print(f"nl_cond: {rule.nl_cond}")
        print("-" * 100 + "\n" * 5)
        rule_with_cond.append(
            {
                "text": rule_obj.text,
                "logic_cond": rule_obj.logic_cond,
                "nl_cond": rule_obj.nl_cond,
                "sources": rule_obj.sources,
                "weight": rule_obj.weight,
            }
        )
        with open(output_file, "w") as f:
            json.dump(rule_with_cond, f, indent=2)
