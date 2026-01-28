"""
1. Execute the GT SQLs and the generated SQLs on the database
2. Compare the generated SQL with GT SQL when the execution results are different
"""

from database import Database
from parser import SQLCollection
from memory import Memory
from loader import load_data, load_preds
from selection import (
    syntax_level_selection,
    group_sql_nodes,
    intra_group_selection,
)
from analyzer import collect_schema_info
from prompts import get_simple_comparison_prompt
import argparse
from typing import Tuple
import time
import os
import json
from collections import defaultdict
from prompts import get_simple_comparison_prompt
from llm_infer import llm_check
from utils import parse_json

verifier = os.getenv("MODEL_ABBR")


def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="birddev")
    parser.add_argument("--method_name", type=str, default="alphasql")
    parser.add_argument("--model_name", type=str, default="7B")
    parser.add_argument("--selector", type=str, default="intent_1229")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_option()
    qid_info = load_data(args.dataset_name)
    qid_pred, qid_sql_cnt = load_preds(
        args.method_name, args.dataset_name, args.model_name
    )
    print(f"len(qid_info): {len(qid_info)}")
    print(f"len(qid_pred): {len(qid_pred)}")
    print(f"len(qid_sql_cnt): {len(qid_sql_cnt)}")
    output_dir = f"results/{args.dataset_name}/{args.method_name}/{args.model_name}"
    eval_dir = f"eval_results/{args.dataset_name}/{args.method_name}/{args.model_name}"
    # if the SQLs in each group are correct, we can skip the comparison
    # only compare the SQLs when the execution results are different
    qid_sql_acc_file = f"eval_results/{args.dataset_name}/{args.method_name}/{args.model_name}/gp_sql_acc.json"
    qid_sql_acc = json.load(open(qid_sql_acc_file, "r"))
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{args.selector}.json")
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            results = json.load(f)
    else:
        results = {}
    print(f"processed {len(results)} questions")

    db_memory = Memory()
    for qid, gp_sql_acc_list in qid_sql_acc.items():
        if int(qid) not in qid_info:
            continue
        info = qid_info[int(qid)]
        question = info["question"]
        evidence = info["evidence"]
        db_name = info["db_id"]
        gt_sql = info["SQL"]
        if type(gt_sql) == list:
            continue
        preds = []
        for sql_acc_dict in gp_sql_acc_list:
            sqls = sql_acc_dict["sqls"]
            acc = sql_acc_dict["acc1"]
            if acc == 1:
                continue
            preds.extend(sqls)
        if len(preds) == 0:
            continue
        preds = preds + [gt_sql]
        if db_name not in db_memory.memory:
            db = Database(args.dataset_name, db_name)
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
        start_time = time.time()
        sql_collection = SQLCollection(preds, db)
        # if all the sqls have the same execution results, skip the selection
        exec_res_set = set()
        for sql, exec_res in sql_collection.exe_results.items():
            if exec_res == "Time Out" or exec_res == "Unexecutable":
                continue
            if len(exec_res) == 0 or len(exec_res) == 1 and len(exec_res[0]) == 0:
                continue
            exec_res_set.add(frozenset(exec_res))

        if len(exec_res_set) <= 1:
            if len(sql_collection.sqls) == 0:
                selected_sql = "Error SQL"
            else:
                selected_sql = sql_collection.sqls[0]
            results[qid] = {
                "sql_logs": [],
                "selected_sql": selected_sql,
                "time_cost": time.time() - start_time,
            }
            continue
        print(f"\n\n[QID]: {qid}, [DB Name]: {db_name}")
        print(f"[Question]: {question}")
        print("=====Syntax Level Selection=====")

        sql_nodes = syntax_level_selection(sql_collection, question, evidence)
        gt_sql_node = None
        gen_sql_nodes = []
        for sql_node in sql_nodes:
            if sql_node.org_sql == gt_sql:
                gt_sql_node = sql_node
            else:
                gen_sql_nodes.append(sql_node)
        if len(gen_sql_nodes) == 0 or gt_sql_node is None:
            continue
        original_size = len(preds)
        syntax_filtered_size = len(sql_nodes)
        print(f"syntax_filtered: {original_size} -> {syntax_filtered_size}")
        # keep only the top 3 groups of sqls with the same execution results

        print("=====Majority Voting Selection=====")
        sql_cnt = qid_sql_cnt.get(qid, {})
        grouped_sql_nodes, filtered_group_cnt = group_sql_nodes(
            gen_sql_nodes, sql_cnt, filtering=False
        )
        print(f"filtered groups: {len(grouped_sql_nodes)}")
        # select the best sql from each group
        print("=====Intra Group Selection=====")
        intra_group_selected_sqls = intra_group_selection(
            sql_collection, question, evidence, grouped_sql_nodes
        )
        schema_note = collect_schema_info(
            sql_collection.table_columns, sql_collection.db
        )
        # Comparison between the GT SQL and the generated SQLs
        prompts = []
        for gen_sql_node in intra_group_selected_sqls:
            if gen_sql_node.org_sql == gt_sql:
                continue
            prompt = get_simple_comparison_prompt(
                question, evidence, schema_note, gt_sql_node, gen_sql_node
            )
            prompts.append(prompt)
        responses = llm_check(prompts)
        result = dict(info)
        result["time_cost"] = time.time() - start_time
        result["schema"] = sql_collection.table_columns
        sql_logs = []
        for sql_node in [gt_sql_node] + intra_group_selected_sqls:
            sql_logs.append(
                {
                    "sql": sql_node.org_sql,
                    "notes": sql_node.notes,
                }
            )
        result["sql_logs"] = sql_logs
        comparison_notes = []
        for i, response in enumerate(responses):
            note = parse_json(response)
            try:
                correctness = int(note["better_sql"] == "SQL1")
            except:
                if "better_sql" in note:
                    index = note["better_sql"].rindex("better_sql")
                    if "SQL1" in note["better_sql"][index:]:
                        correctness = 1
                    else:
                        correctness = 0
                else:
                    correctness = 0.5
            comparison_notes.append(
                {
                    "SQL1": gt_sql_node.org_sql,
                    "SQL2": intra_group_selected_sqls[i].org_sql,
                    "comparison_note": note,
                    "correctness": correctness,
                }
            )
        result["comparison_notes"] = comparison_notes
        results[qid] = result

        with open(output_file, "w") as f:
            json.dump(results, f, indent=4)
