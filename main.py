from database import Database
from parser import SQLCollection
from memory import Memory
from loader import load_data, load_preds
from selection import (
    collective_selection,
    pointwise_selection,
    rank_sql_nodes,
    exhaustive_pairwise,
    filter_by_returned_columns,
)
import argparse
from typing import Tuple
import time
import os
import json
from collections import defaultdict

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

    db_memory = Memory()
    flag = False
    for qid, preds in qid_pred.items():
        # if int(qid) not in [352, 423, 433, 479, 507, 685, 694, 959, 1009, 1241, 1510]:
        #     continue
        if str(qid) in results:
            continue
        if qid not in qid_to_be_checked:
            continue
        info = qid_info[qid]
        question = info["question"]
        evidence = info["evidence"]
        db_name = info["db_id"]
        # [DEBUG]: compare GT SQL with the predicted sqls
        gt_sql = info["SQL"]
        if len(gt_sql) > 1:
            continue
        preds = [gt_sql[0]] + preds
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
            # print("=====joinable columns=====")
            # for key, values in db.joinable_columns.items():
            #     print(f"{key}: {values}")
            # print(db.display())
            # print("\n" * 5)
        else:
            db = db_memory.get(db_name)
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
            continue
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
        # # print("=====Filtering by Returned Columns=====")
        # sql_nodes = filter_by_returned_columns(sql_collection, question, evidence)
        print("=====Pointwise Selection=====")
        sql_nodes = pointwise_selection(sql_collection, question, evidence)
        # filtered_sql_nodes = []
        # tb_col_nodes = defaultdict(list)
        # for sql_node in sql_nodes:
        #     tables, columns = sql_node.tables, sql_node.columns
        #     tables = [tb for tb in tables if tb in db.tables]
        #     filtered_columns = []
        #     for col in columns:
        #         col = col.split(".", maxsplit=1)[-1]
        #         for tb in tables:
        #             if col in db.tables[tb].columns:
        #                 filtered_columns.append(col)
        #                 break
        #     sql_cnt = qid_sql_cnt.get(qid, {})
        #     print(
        #         f"SQL: {sql_node.org_sql}\nTables: {tables}\nColumns: {filtered_columns}"
        #         f"\nORG Columns: {sql_node.columns}"
        #         f"\nNum: {sql_cnt.get(sql_node.org_sql, 1)}"
        #         f"\nAcc: {sql_acc_dict.get(sql_node.org_sql, 0)}"
        #     )
        #     print("-" * 100)
        #     tb_col = frozenset(tables + filtered_columns)
        #     tb_col_nodes[tb_col].append(sql_node)
        # for tb_col, nodes in tb_col_nodes.items():
        #     if len(nodes) > 1 or sql_cnt.get(nodes[0].org_sql, 1) > 1:
        #         filtered_sql_nodes.extend(nodes)
        # sql_nodes = filtered_sql_nodes
        sorted_sql_nodes = rank_sql_nodes(sql_nodes, qid_sql_cnt[qid])
        print("=====Collective Selection=====")
        selected_sql, comparison_notes = collective_selection(
            sql_collection, question, evidence, sorted_sql_nodes
        )
        # selected_sql, comparison_notes = exhaustive_pairwise(
        #     sql_collection, question, evidence, sql_nodes
        # )
        # selected_sql = None
        # comparison_notes = []
        print("\n" * 5)
        sql_logs = []
        for sql_node in sorted_sql_nodes:
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
                    "rank": sql_node.rank,
                    "acc": sql_acc_dict.get(sql_node.org_sql, 0),
                }
            )
        results[qid] = {
            "selected_sql": selected_sql,
            "selected_acc": sql_acc_dict.get(selected_sql, 0),
            "time_cost": time.time() - start_time,
            "sql_logs": sql_logs,
            "comparison_notes": comparison_notes,
        }
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
