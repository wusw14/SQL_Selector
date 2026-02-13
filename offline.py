"""
Ask the LLM to generate rules for the comparison between the GT SQL and the generated SQLs
Only generate rules for the incorrect cases
"""

from database import Database
from parser import SQLCollection
from memory import Memory
from loader import load_data, load_preds, load_all_preds
from selection import syntax_level_selection, group_sql_nodes, intra_group_selection
import argparse
from typing import Tuple, Dict, Any, List
from parser import SQLCollection, SQLNode
import time
import os
import json
from collections import defaultdict
from copy import deepcopy
from representation import Representation, get_relevant_rules
import numpy as np
from binary_comparison import binary_comparison
from rule_gen import rule_gen

verifier = os.getenv("MODEL_ABBR")


def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="birddev")
    parser.add_argument("--method_name", type=str, default="alphasql")
    parser.add_argument("--num_iterations", type=int, default=1)
    parser.add_argument("--rule_mode", type=str, default="relevant")
    return parser.parse_args()


def preprocess_preds(
    qid_preds: Dict[int, List[str]],
    qid_info: Dict[int, Dict[str, Any]],
    qid_sql_acc: Dict[int, Dict],
) -> Dict[int, SQLCollection]:
    qid_sql_collections = {}
    db_memory = Memory()
    skipped_qids = []
    for qid, preds in qid_preds.items():
        if qid not in qid_info:
            continue
        info = qid_info[qid]
        # if all the preds are correct, skip
        sql_acc = qid_sql_acc[qid]
        if all(sql_acc.get(sql, 0) == 1 for sql in preds):
            skipped_qids.append(qid)
            print(f"Skipped QID: {qid} because all the preds are correct")
            continue
        question = info["question"]
        evidence = info["evidence"]
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
        print(f"[QID]: {qid}, [DB Name]: {db_name}")
        # add GT SQL to the preds
        if type(info["SQL"]) == list:
            gt_sql = info["SQL"][0]
        else:
            gt_sql = info["SQL"]
        preds.append(gt_sql)
        sql_collection = SQLCollection(preds, db, info)
        for sql_node in sql_collection.sql_nodes:
            sql_node.acc = sql_acc.get(sql_node.org_sql, 0)
        # syntax level selection
        sql_nodes = syntax_level_selection(sql_collection, question, evidence)
        # intra-group selection
        grouped_sql_nodes, _ = group_sql_nodes(sql_nodes, {}, filtering=False)
        intra_group_selected_sqls = intra_group_selection(
            sql_collection, question, evidence, grouped_sql_nodes
        )
        sql_collection.intra_selected_sql_nodes = intra_group_selected_sqls
        # split the sql_nodes into two parts: the correct ones and the incorrect ones
        gt_sql_nodes, incorrect_sql_nodes = [], []
        for sql_node in sql_nodes:
            if sql_node.org_sql == gt_sql:
                gt_sql_nodes.append(sql_node)
        if len(gt_sql_nodes) == 0:
            print(f"qid: {qid}, gt_sql: {gt_sql}")
            print("skipped!!!")
            continue
        for sql_node in intra_group_selected_sqls:
            if sql_acc.get(sql_node.org_sql, 0) == 0 and sql_node not in gt_sql_nodes:
                incorrect_sql_nodes.append(sql_node)
        sql_collection.gt_sql_nodes = gt_sql_nodes
        sql_collection.incorrect_sql_nodes = deepcopy(incorrect_sql_nodes)
        qid_sql_collections[qid] = sql_collection
    print(f"len(qid_sql_collections): {len(qid_sql_collections)}")
    print(f"len(skipped_qids): {len(skipped_qids)}")
    return qid_sql_collections


def write_qid_sql_collections(
    qid_sql_collections: Dict[int, SQLCollection],
    t: int,
    args,
    step="rule_gen",
    tgt_db_id=None,
):
    output_dir = f"results/{args.dataset_name}/{args.method_name}/iterative_rules"
    if tgt_db_id is not None:
        output_dir = os.path.join(output_dir, tgt_db_id)
    os.makedirs(output_dir, exist_ok=True)
    if step == "comparison":
        output_file = os.path.join(output_dir, f"t={t}.1.json")
    else:
        output_file = os.path.join(output_dir, f"t={t}.json")
    qid_results = {}
    for qid, sql_collection in qid_sql_collections.items():
        result = sql_collection.info
        result["rules"] = sql_collection.rules
        result["comparison_notes"] = sql_collection.comparison_notes

        qid_results[qid] = result
    with open(output_file, "w") as f:
        json.dump(qid_results, f, indent=2)


if __name__ == "__main__":
    args = parse_option()
    # qids = [273, 432, 451, 590, 628]
    qid_info = load_data(args.dataset_name)
    db_qids = defaultdict(list)
    for qid, info in qid_info.items():
        db_qids[info["db_id"]].append(qid)
    tgt_db_id = "full_dev"
    # # [DEBUG]test on one database
    # qids = [
    #     qid
    #     for qid, info in qid_info.items()
    #     if info["db_id"] in ["superhero", "card_games"]
    # ]
    # qid_info = {qid: info for qid, info in qid_info.items() if qid in qids}
    print(f"len(qid_info): {len(qid_info)}")
    # Step 1: load all the predictions from all the LLM predictions
    qid_preds, qid_sql_acc = load_all_preds(args)

    # Step 2: preprocess the preds into sql_collections
    qid_sql_collections = preprocess_preds(qid_preds, qid_info, qid_sql_acc)

    # Step 3: iteratively generate the rules for incorrect cases
    qid_early_stop_cnt = defaultdict(int)
    qid_preds_with_rules = qid_preds
    for t in range(args.num_iterations):
        representation_model = Representation(qid_preds_with_rules)
        print(f"=====Iteration {t}=====")
        output_file = f"results/{args.dataset_name}/{args.method_name}/iterative_rules/{tgt_db_id}/t={t + 1}.json"
        if os.path.exists(output_file):
            # recover the comparison notes and the rules from the file
            qid_sql_collections_js = json.load(open(output_file, "r"))
            qid_preds_with_rules = {}
            for qid_str, result in qid_sql_collections_js.items():
                if int(qid_str) not in qid_sql_collections:
                    continue
                rules = result["rules"]
                comparison_notes = result["comparison_notes"]
                qid_sql_collections[int(qid_str)].rules = rules
                qid_sql_collections[int(qid_str)].comparison_notes = comparison_notes
                if len(rules) > 0:
                    qid_preds_with_rules[int(qid_str)] = qid_preds[int(qid_str)]
            continue
        rule_mode = "none" if t == 0 else args.rule_mode
        print(f"rule_mode: {rule_mode}")
        # print(f"len(qids): {len(qids)}")
        # do the binary comparison
        qid_compare_results = binary_comparison(
            qids, qid_sql_collections, rule_mode, representation_model, db_qids
        )
        for qid, compare_result in qid_compare_results.items():
            qid_sql_collections[qid].comparison_notes = compare_result[
                "comparison_notes"
            ]
        write_qid_sql_collections(
            qid_sql_collections, t + 1, args, step="comparison", tgt_db_id=tgt_db_id
        )
        # generate the rules for the incorrect cases
        qid_sql_collections = rule_gen(qid_compare_results, qid_sql_collections)
        # write the qid_sql_collections to the file
        write_qid_sql_collections(qid_sql_collections, t + 1, args, tgt_db_id=tgt_db_id)
        qids = []
        qid_preds_with_rules = {}
        for qid, sql_collection in qid_sql_collections.items():
            if t + 1 - len(sql_collection.rules) <= 2:
                qids.append(qid)
            if len(sql_collection.rules) > 0:
                qid_preds_with_rules[qid] = qid_preds[qid]
        if len(qids) == 0:
            break
