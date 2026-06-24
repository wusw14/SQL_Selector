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


if __name__ == "__main__":
    args = parse_option()
    qid_info = load_data(args.dataset_name)
    qid_preds, qid_gp_sql_acc = load_all_preds(args)
    qid_db = {qid: info["db_id"] for qid, info in qid_info.items()}

    rule_file = (
        f"results/{args.dataset_name}/alphasql/iterative_rules/full_dev/t=1.json"
    )
    qid_rules = json.load(open(rule_file, "r"))
    qid_rules = {
        int(qid): item["rules"][-1]
        for qid, item in qid_rules.items()
        if len(item["rules"]) > 0
    }
    output_file = (
        f"results/{args.dataset_name}/alphasql/iterative_rules/full_dev/score_v2.json"
    )

    db_memory = Memory()
    flag = False
    processed_qids = []
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            rule_score_collection = json.load(f)
        for item in rule_score_collection:
            processed_qids.append(item["source"])
        processed_qids = list(set(processed_qids))
    else:
        rule_score_collection = []
    print(f"processed_qids: {len(processed_qids)}")

    for qid, rules in qid_rules.items():
        if qid in processed_qids:
            continue
        if qid not in qid_info or qid not in qid_gp_sql_acc:
            continue
        gp_sql_acc = qid_gp_sql_acc[qid]
        info = qid_info[qid]
        question = info["question"]
        evidence = info["evidence"]
        gt_sqls = info["SQL"]
        db_name = info["db_id"]
        if db_name not in db_memory.memory:
            db = Database(args.dataset_name, db_name)
            db_memory.add(db_name, db)
            print("=====joinable columns=====")
            joinable_column_sets = set()
            for key, values in db.joinable_columns.items():
                joinable_column_sets.add(frozenset(values))
            for column_set in joinable_column_sets:
                print(f"{column_set}")
            print("\n" * 5)
        else:
            db = db_memory.get(db_name)
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
        preds = correct_sqls + incorrect_sqls
        print(f"size of preds: {len(preds)}")
        # preds = qid_preds[qid]
        start_time = time.time()
        sql_collection = SQLCollection(preds, db, info)
        sql_node_votes = rule_based_selection(
            sql_collection, sql_collection.sql_nodes, question, evidence, rules
        )
        # # usefulness evaluation
        rule_pos, rule_neg = defaultdict(list), defaultdict(list)
        for sql_node in sql_collection.sql_nodes:
            for i, rule in enumerate(rules):
                score = sql_node.score_each_rule[i]
                if sql_node.org_sql in correct_sqls:
                    rule_pos[rule].append(score)
                elif sql_node.org_sql in incorrect_sqls:
                    rule_neg[rule].append(score)
        for i, rule in enumerate(rules):
            pos_scores = rule_pos.get(rule, [1])
            neg_scores = rule_neg.get(rule, [0])
            score_diff = np.mean(pos_scores) - np.mean(neg_scores)
            print(f"rule: {rule}, score_diff: {score_diff:.2f}")
            sql_collection.rule_scores[rule] = {
                "pos_scores": pos_scores,
                "neg_scores": neg_scores,
            }
            rule_score_collection.append(
                {
                    "rule": rule,
                    "source": qid,
                    "question": question,
                    "pos_scores": pos_scores,
                    "neg_scores": neg_scores,
                    "score_diff": score_diff,
                    "time_cost": time.time() - start_time,
                }
            )
            with open(output_file, "w") as f:
                json.dump(rule_score_collection, f, indent=4)
        print(f"time cost: {time.time() - start_time:.2f}s")
