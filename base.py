from database import Database
from memory import Memory
from loader import load_data, load_preds
from utils import execute_sql_wrapper, execute_sql
from parser import SQLCollection
from selection import collective_selection, pointwise_selection
import argparse
from typing import Tuple
import time
import os
import json

verifier = os.getenv("MODEL_ABBR")


def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="birddev")
    parser.add_argument("--method_name", type=str, default="alphasql")
    parser.add_argument("--model_name", type=str, default="7B")
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
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "exec.json")
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            results = json.load(f)
    else:
        results = {}
    print(f"processed {len(results)} questions")

    eval_base = json.load(open(os.path.join(eval_dir, "majority.json"), "r"))
    qid_to_be_checked = []
    for qid, res in eval_base.items():
        if res["upper_acc"] == 0 or res["lower_acc"] == 1:
            continue
        qid_to_be_checked.append(int(qid))
    print(f"len(qid_to_be_checked): {len(qid_to_be_checked)}")

    db_memory = Memory()
    flag = False
    for qid, preds in qid_pred.items():
        if qid not in qid_info:
            continue
        if str(qid) in results:
            continue
        if qid not in qid_to_be_checked:
            continue
        info = qid_info[qid]
        question = info["question"]
        evidence = info["evidence"]
        db_name = info["db_id"]
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
            exec_res_set.add(frozenset(exec_res))

        if len(exec_res_set) <= 1:
            if len(sql_collection.sqls) == 0:
                selected_sql = "Error SQL"
            else:
                selected_sql = sql_collection.sqls[0]
            results[qid] = {
                "sql_logs": [{"sql": selected_sql}],
                "selected_sql": selected_sql,
                "time_cost": time.time() - start_time,
            }
            continue
        print(f"[QID]: {qid}, [DB Name]: {db_name}")
        print(f"[Question]: {question}")
        sql_logs = []
        for sql in sql_collection.sqls:
            sql_logs.append({"sql": sql})
        results[qid] = {
            "sql_logs": sql_logs,
            "selected_sql": sql_logs[0]["sql"],
            "time_cost": time.time() - start_time,
        }
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
