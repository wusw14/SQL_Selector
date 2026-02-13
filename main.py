from database import Database
from parser import SQLCollection
from memory import Memory
from loader import load_data, load_preds, load_all_preds
from selection import (
    collective_selection,
    syntax_level_selection,
    rank_sql_nodes,
    exhaustive_pairwise,
    filter_by_returned_columns,
    group_sql_nodes,
    intra_group_selection,
    inter_group_selection,
)
from analyzer import classify_query_type
import argparse
from typing import Tuple
import time
import os
import json
from collections import defaultdict
from representation import Representation, get_relevant_rules
import numpy as np

verifier = os.getenv("MODEL_ABBR")


def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="birddev")
    parser.add_argument("--method_name", type=str, default="alphasql")
    parser.add_argument("--model_name", type=str, default="7B")
    parser.add_argument("--selector", type=str, default="intent_1229")
    parser.add_argument(
        "--rule_mode",
        type=str,
        choices=["none", "random", "relevant", "ideal"],
        default="relevant",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_option()
    qid_info = load_data(args.dataset_name)
    qid_info = {
        qid: info
        for qid, info in qid_info.items()
        if info["db_id"] == "california_schools"
    }
    qid_pred, qid_sql_cnt = load_preds(
        args.method_name, args.dataset_name, args.model_name
    )
    rule_qid_preds, _ = load_all_preds(args)
    rule_file = "results/birddev/alphasql/iterative_rules/superhero+card_games/t=3.json"
    qid_rules = json.load(open(rule_file, "r"))
    qid_rules = {
        int(qid): item["rules"][-1]
        for qid, item in qid_rules.items()
        if len(item["rules"]) > 0
    }
    rule_qid_preds = {
        qid: preds for qid, preds in rule_qid_preds.items() if qid in qid_rules
    }
    representation_model = Representation(rule_qid_preds)
    query_vectors = representation_model.query_vectors
    print(f"len(query_vectors): {len(query_vectors)}")
    print(f"len(qid_info): {len(qid_info)}")
    print(f"len(qid_pred): {len(qid_pred)}")
    print(f"len(qid_sql_cnt): {len(qid_sql_cnt)}")
    output_dir = f"results/{args.dataset_name}/{args.method_name}/{args.model_name}"
    eval_dir = f"eval_results/{args.dataset_name}/{args.method_name}/{args.model_name}"
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

    eval_base = json.load(open(os.path.join(eval_dir, "exec.json"), "r"))
    qid_to_be_checked = []
    for qid, res in eval_base.items():
        if res["upper_acc"] == 0 or res["lower_acc"] == 1:
            continue
        qid_to_be_checked.append(int(qid))
    print(f"len(qid_to_be_checked): {len(qid_to_be_checked)}")

    db_qids = defaultdict(list)
    for qid, info in qid_info.items():
        db_qids[info["db_id"]].append(qid)

    db_memory = Memory()
    flag = False
    for qid, preds in qid_pred.items():
        # if int(qid) not in [352, 423, 433, 479, 507, 685, 694, 959, 1009, 1241, 1510]:
        #     continue
        if str(qid) in results:
            continue
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
        start_time = time.time()
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
        sql_collection = SQLCollection(preds, db, info)
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
        # keep only the top 3 groups of sqls with the same execution results
        print("=====Majority Voting Selection=====")
        sql_cnt = qid_sql_cnt.get(qid, {})
        grouped_sql_nodes, filtered_group_cnt = group_sql_nodes(sql_nodes, sql_cnt)
        print(f"filtered groups: {len(grouped_sql_nodes)}")
        if len(grouped_sql_nodes) <= 1:
            selected_sql = sql_nodes[0].org_sql
            results[qid] = {
                "selected_sql": selected_sql,
                "selected_acc": sql_acc_dict.get(selected_sql, 0),
                "time_cost": time.time() - start_time,
                "sql_logs": [],
                "comparison_notes": [],
            }
            continue
        # select the best sql from each group
        print("=====Intra Group Selection=====")
        intra_group_selected_sqls = intra_group_selection(
            sql_collection, question, evidence, grouped_sql_nodes
        )
        print("=====Inter Group Selection=====")
        rules = get_relevant_rules(
            qid,
            qid_rules,
            preds,
            representation_model,
            args.rule_mode,
            top_k=5,
            exclude_qids=db_qids[db_name],
        )
        # print("[DEBUG][rule_dict]")
        # for rule_category, rules in rule_dict.items():
        #     print(f"[DEBUG][rule_category]: {rule_category}")
        #     print(f"[DEBUG][rules]: {rules}")
        sql_node_votes, comparison_notes = inter_group_selection(
            sql_collection,
            question,
            evidence,
            intra_group_selected_sqls,
            rules,
            args.rule_mode,
        )
        print("=====Final Selection=====")
        for sql_node in sql_node_votes:
            for gp, cnt in zip(grouped_sql_nodes, filtered_group_cnt):
                if sql_node in gp:
                    sql_node_votes[sql_node] += cnt * 1e-3
                    break
        sorted_sql_node_votes = sorted(
            sql_node_votes.items(), key=lambda x: x[1], reverse=True
        )
        selected_sql_node = sorted_sql_node_votes[0][0]
        selected_sql = selected_sql_node.org_sql
        # selected_sql, comparison_notes = exhaustive_pairwise(
        #     sql_collection, question, evidence, sql_nodes
        # )
        # selected_sql = None
        # comparison_notes = []
        print("\n" * 5)
        sql_logs = []
        for sql_node, vote in sorted_sql_node_votes:
            # if min_warning_cnt is None or sql_node.warning_cnt < min_warning_cnt:
            #     min_warning_cnt = sql_node.warning_cnt
            #     selected_sql = sql_node.org_sql
            exec_stats = {
                "rows": len(sql_node.exec_res),
                "columns": len(sql_node.exec_res[0]),
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
                }
            )
        result = dict(info)
        result["selected_sql"] = selected_sql
        result["selected_acc"] = sql_acc_dict.get(selected_sql, 0)
        result["time_cost"] = time.time() - start_time
        result["sql_logs"] = sql_logs
        result["comparison_notes"] = comparison_notes
        results[qid] = result
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
