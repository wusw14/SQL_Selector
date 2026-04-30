from database import Database
from parser import SQLCollection
from memory import Memory
from loader import load_data, load_preds, load_all_preds
from selection import (
    syntax_level_selection,
    group_sql_nodes,
    intra_group_selection,
    inter_group_selection,
    rule_based_selection,
    final_adjustment,
)
import argparse
from typing import Tuple
import time
import os
import json
from collections import defaultdict
from representation import Representation, get_relevant_rules, get_representation
import numpy as np
from rule_lib import load_rule_collection
import random

verifier = os.getenv("MODEL_ABBR")


def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="birddev")
    parser.add_argument("--method_name", type=str, default="alphasql")
    parser.add_argument("--model_name", type=str, default="Qwen2.5-7B")
    parser.add_argument("--selector", type=str, default="agentar")
    parser.add_argument(
        "--rule_mode",
        type=str,
        choices=["none", "random", "relevant", "ideal"],
        default="none",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_option()
    qid_info = load_data(args.dataset_name)
    # set seed
    random.seed(42 + int(args.selector.split("_")[-1][-1]))
    # get results based on the results of chase.json
    output_dir = f"results/{args.dataset_name}/{args.method_name}/{args.model_name}"
    eval_dir = f"eval_results/{args.dataset_name}/{args.method_name}/{args.model_name}"
    qid_sql_acc_file = f"eval_results/{args.dataset_name}/{args.method_name}/{args.model_name}/gp_sql_acc.json"
    qid_sql_acc = json.load(open(qid_sql_acc_file, "r"))
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{args.selector}.json")
    chase_file = os.path.join(output_dir, "chase.json")
    chase_results = json.load(open(chase_file, "r"))
    results = {}
    for qid, result in chase_results.items():
        results[qid] = result
        # select one sql from each group
        gp_sql_acc = qid_sql_acc.get(qid, {})
        if len(gp_sql_acc) == 0 or "comparison_notes" not in result:
            continue
        sample_sqls = []
        for gp_sql in gp_sql_acc:
            sqls = gp_sql["sqls"]
            acc = gp_sql["acc1"]
            # randomly sample one sql from sqls
            sql = random.choice(sqls)
            sample_sqls.append(sql)
        sql_pair_score = {}
        comparison_notes = result["comparison_notes"]
        for comparison_note in comparison_notes:
            sql1 = comparison_note["SQL1"]
            sql2 = comparison_note["SQL2"]
            note = comparison_note["comparison_note"]
            if "better_sql" not in note:
                score = 0.5
            else:
                index = note.rindex("better_sql")
                if "SQL1" in note[index:]:
                    score = 1
                elif "SQL2" in note[index:]:
                    score = 0
                else:
                    score = 0.5
            sql_pair_score[(sql1, sql2)] = score
        # for (sql1, sql2), score in sql_pair_score.items():
        #     print(sql1)
        #     print(sql2)
        #     print(score)
        #     print("-" * 100)
        # sum up votes for each sql
        sql_votes = {}
        for sql1 in sample_sqls:
            sql_votes[sql1] = 0
            for sql2 in sample_sqls:
                if sql1 == sql2:
                    continue
                sql_votes[sql1] += sql_pair_score.get((sql1, sql2), 0.5)
        # sorted sql
        sql_votes = sorted(sql_votes.items(), key=lambda x: x[1], reverse=True)
        selected_sql = sql_votes[0][0]
        result["selected_sql"] = selected_sql
        result["comparison_notes"] = []
        sql_logs = []
        for sql, vote in sql_votes:
            sql_logs.append({"sql": sql, "vote": vote})
        result["sql_logs"] = sql_logs
        results[qid] = result
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
