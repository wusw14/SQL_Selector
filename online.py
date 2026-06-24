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

verifier = os.getenv("MODEL_ABBR")


# NL, SQL, execution results
rules = [
    "Adhere strictly to instructions in the evidence, particularly regarding return values and metric calculations.",
    "Ensure the filters fully match the natural language query—no missing or extra conditions.",
    "For queries with potential ties, ensure SQL returns all tied records (e.g., use RANK() over ROW_NUMBER(), use max/min over order by xxx limit 1).",
    "Always group by unique identifiers (e.g., ID) instead of non-unique attributes (e.g., name) to ensure accurate aggregation and avoid merging distinct entities.",
    "Exclude sentinel values (e.g., NULL, zero) from numeric aggregations when they represent missing or invalid data.",
]


def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="birddev")
    parser.add_argument("--method_name", type=str, default="alphasql")
    parser.add_argument("--model_name", type=str, default="Qwen2.5-7B")
    parser.add_argument("--selector", type=str, default="intent_1229")
    parser.add_argument(
        "--rule_mode",
        type=str,
        choices=["none", "random", "relevant", "ideal"],
        default="relevant",
    )
    parser.add_argument(
        "--retrieval_mode",
        type=str,
        choices=["crosscase", "crossdb", "crossds"],
        default="crosscase",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_option()
    qid_info = load_data(args.dataset_name)
    qid_pred, qid_sql_cnt = load_preds(
        args.method_name, args.dataset_name, args.model_name
    )
    qid_db = {qid: info["db_id"] for qid, info in qid_info.items()}
    rule_file = "alphasql/iterative_rules/full_dev/rule_with_cond_v4.json"
    if args.retrieval_mode != "crossds":
        rule_file = f"../verifier/results/{args.dataset_name}/{rule_file}"
    else:
        if args.dataset_name == "birddev":
            rule_file = "../verifier/results/spidertest/{rule_file}"
        else:
            rule_file = "../verifier/results/birddev/{rule_file}"
    rule_collection = load_rule_collection(rule_file, qid_db)
    print(f"len(qid_info): {len(qid_info)}")
    print(f"len(qid_pred): {len(qid_pred)}")
    print(f"len(qid_sql_cnt): {len(qid_sql_cnt)}")
    output_dir = f"results/{args.dataset_name}/{args.method_name}/{args.model_name}"
    eval_dir = f"../verifier/eval_results/{args.dataset_name}/{args.method_name}/{args.model_name}"
    qid_sql_acc_file = f"../verifier/eval_results/{args.dataset_name}/{args.method_name}/{args.model_name}/gp_sql_acc.json"
    qid_sql_acc = json.load(open(qid_sql_acc_file, "r"))
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{args.selector}.json")
    if os.path.exists(output_file) and "debug" not in args.selector:
        with open(output_file, "r") as f:
            results = json.load(f)
    else:
        results = {}
    print(f"processed {len(results)} questions")

    eval_base = json.load(open(os.path.join(eval_dir, "majority.json"), "r"))
    qid_to_be_checked = []
    for qid, res in eval_base.items():
        if res["upper_acc"] == 0 or res["lower_acc"] == 1 or (qid in results):
            continue
        qid_to_be_checked.append(int(qid))
    print(f"len(qid_to_be_checked): {len(qid_to_be_checked)}")

    db_qids = defaultdict(list)
    for qid, info in qid_info.items():
        db_qids[info["db_id"]].append(qid)

    db_memory = Memory()
    flag = False
    for qid, preds in qid_pred.items():
        if qid not in qid_to_be_checked:
            continue
        if qid not in qid_info:
            continue
        info = qid_info[qid]
        question = info["question"]
        evidence = info["evidence"]
        db_name = info["db_id"]
        # [DEBUG]: compare GT SQL with the predicted sqls
        gt_sql = info["SQL"]
        # if db_name != "california_schools":
        #     continue
        gp_acc_list = qid_sql_acc[str(qid)]
        sql_acc_dict = {}
        for sql_acc in gp_acc_list:
            sqls = sql_acc["sqls"]
            acc = sql_acc["acc1"]
            for sql in sqls:
                sql_acc_dict[sql] = acc
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
        sql_collection = SQLCollection(preds, db, info)
        # if all the sqls have the same execution results, skip the selection
        exec_res_set = set()
        for sql, exec_res in sql_collection.exe_results.items():
            if exec_res == "Time Out" or exec_res == "Unexecutable":
                continue
            if len(exec_res) == 0 or len(exec_res) == 1 and len(exec_res[0]) == 0:
                continue
            exec_res_set.add(frozenset(exec_res))

        result = dict(info)
        if len(exec_res_set) <= 1:
            if len(sql_collection.sqls) == 0:
                selected_sql = preds[0]
            else:
                selected_sql = sql_collection.sqls[0]
            result["selected_sql"] = selected_sql
            result["selected_acc"] = sql_acc_dict.get(selected_sql, 0)
            result["time_cost"] = time.time() - start_time
            results[str(qid)] = result
            continue
        print(f"[QID]: {qid}, [DB Name]: {db_name}")
        print(f"[Question]: {question}")
        # query_type = classify_query_type(question)
        # # print("=====Filtering by Returned Columns=====")
        # sql_nodes = filter_by_returned_columns(sql_collection, question, evidence)
        print("=====Syntax Level Selection=====")
        sql_nodes = syntax_level_selection(sql_collection, question, evidence)
        if len(sql_nodes) == 0:
            selected_sql = preds[0]
            results[qid] = {
                "selected_sql": selected_sql,
                "selected_acc": sql_acc_dict.get(selected_sql, 0),
                "time_cost": time.time() - start_time,
                "sql_logs": [],
                "comparison_notes": [],
            }
            continue
        original_size = len(preds)
        syntax_filtered_size = len(sql_nodes)
        print(f"syntax_filtered: {original_size} -> {syntax_filtered_size}")

        print("=====Grouping=====")
        sql_cnt = qid_sql_cnt.get(qid, {})
        grouped_sql_nodes, filtered_group_cnt = group_sql_nodes(
            sql_nodes, sql_cnt, filtering=False
        )
        print(f"filtered groups: {len(grouped_sql_nodes)}")
        if len(grouped_sql_nodes) <= 1:
            selected_sql = sql_nodes[0].org_sql
            result["selected_sql"] = selected_sql
            result["selected_acc"] = sql_acc_dict.get(selected_sql, 0)
            result["time_cost"] = time.time() - start_time
            results[str(qid)] = result
            continue

        syntax_time = time.time() - start_time

        print("=====Rule-based Selection=====")
        all_sqls = [sql_node.org_sql for sql_node in sql_nodes]
        gp_sqls = [gp_sql_nodes[0].org_sql for gp_sql_nodes in grouped_sql_nodes]
        rules, weights = rule_collection.retrieve_relevant(
            gp_sqls, qid, question, evidence, args.retrieval_mode
        )
        print(f"[DEBUG] rules: {rules}")
        print(f"[DEBUG] weights: {weights}")
        rule_retrieval_time = time.time() - start_time - syntax_time
        sql_node_votes = rule_based_selection(
            sql_collection, sql_nodes, question, evidence, rules, weights
        )
        max_vote = max(sql_node_votes.values())
        print(f"[DEBUG] max vote: {max_vote}")
        print(sql_node_votes)
        filtered_sql_nodes = []
        for grouped_nodes in grouped_sql_nodes:
            flag = False
            for sql_node in grouped_nodes:
                if sql_node_votes[sql_node] == max_vote:
                    filtered_sql_nodes.append(sql_node)
                    flag = True
                    break
            if flag == False:
                for sql_node in grouped_nodes:
                    if sql_node_votes[sql_node] >= max_vote - 1:
                        filtered_sql_nodes.append(sql_node)
                        flag = True
                        break
        print(f"[DEBUG] Filtered SQL Nodes: {len(filtered_sql_nodes)}")
        rule_score_time = time.time() - start_time - syntax_time - rule_retrieval_time

        print("=====Pairwise Selection=====")
        # TODO: for those achieves the highest score in rule-based selection
        # For each group, select one SQL
        filtered_sql_node_votes, comparison_notes = inter_group_selection(
            sql_collection,
            question,
            evidence,
            filtered_sql_nodes,
            rules,
            args.rule_mode,
        )
        print(f"[DEBUG] SQL Node Votes: {filtered_sql_node_votes}")
        pairwise_time = (
            time.time()
            - start_time
            - syntax_time
            - rule_retrieval_time
            - rule_score_time
        )

        print("=====Final Selection=====")
        for sql_node, vote in filtered_sql_node_votes.items():
            for gp, cnt in zip(grouped_sql_nodes, filtered_group_cnt):
                if sql_node in gp:
                    sql_node_votes[sql_node] += cnt * 1e-3 + vote
                    break
        sorted_sql_node_votes = sorted(
            sql_node_votes.items(), key=lambda x: x[1], reverse=True
        )
        selected_sql_node = sorted_sql_node_votes[0][0]
        # find the covered sql
        selected_sql_node = final_adjustment(selected_sql_node, sql_nodes)
        selected_sql = selected_sql_node.org_sql
        print("\n" * 5)
        sql_logs = []
        for sql_node, vote in sorted_sql_node_votes:
            # if min_warning_cnt is None or sql_node.warning_cnt < min_warning_cnt:
            #     min_warning_cnt = sql_node.warning_cnt
            #     selected_sql = sql_node.org_sql
            exec_stats = {
                "rows": len(sql_node.exec_res),
                "columns": (
                    len(sql_node.exec_res[0]) if len(sql_node.exec_res) > 0 else 0
                ),
                "exec_time": sql_node.exec_time,
            }
            sql_logs.append(
                {
                    "sql": sql_node.org_sql,
                    "notes": sql_node.notes,
                    "warning_cnt": sql_node.warning_cnt,
                    "exec_stats": exec_stats,
                    "vote": round(vote, 3),
                    "acc": sql_acc_dict.get(sql_node.org_sql, 0),
                    "score_each_rule": sql_node.score_each_rule,
                    "rule_note": sql_node.rule_note,
                }
            )
        result["selected_sql"] = selected_sql
        result["selected_acc"] = sql_acc_dict.get(selected_sql, 0)
        result["time_cost"] = time.time() - start_time
        result["syntax_time"] = syntax_time
        result["rule_retrieval_time"] = rule_retrieval_time
        result["rule_score_time"] = rule_score_time
        result["pairwise_time"] = pairwise_time
        result["rules"] = rules
        result["comparison_notes"] = comparison_notes
        result["sql_logs"] = sql_logs
        results[str(qid)] = result
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        if "debug" in args.selector and len(results) >= 10:
            break
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
