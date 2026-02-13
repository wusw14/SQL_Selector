from database import Database, Table
from parser import SQLCollection
from memory import Memory
from loader import load_data, load_preds
from selection import (
    pointwise_selection,
    if_align_exec_res_with_gt,
    multi_aspect_sql_comparison,
)
from analyzer import collect_schema_info, classify_query_type
import argparse
from typing import Tuple, Dict, List
import time
import os
import json
from collections import defaultdict
from utils import execute_sql_wrapper

verifier = os.getenv("MODEL_ABBR")


def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="birddev")
    parser.add_argument("--method_name", type=str, default="alphasql")
    parser.add_argument("--model_name", type=str, default="7B")
    parser.add_argument("--selector", type=str, default="intent_1229")
    return parser.parse_args()


def find_joinable_columns(tb1: Table, tb2: Table) -> List[Tuple[str, str]]:
    col1_joinable_cols = defaultdict(list)
    for col1_name, col1 in tb1.columns.items():
        if (
            col1.stats["ratio of null values(%)"] > 20
            or col1.stats["number of distinct values"] < 5
        ):
            continue
        for col2_name, col2 in tb2.columns.items():
            if (
                col2.stats["ratio of null values(%)"] > 20
                or col2.stats["number of distinct values"] < 5
            ):
                continue
            # calculate the coverage of the two columns
            sql1 = f"SELECT DISTINCT `{col1_name}` FROM `{tb1.tb_name}`;"
            vals1 = execute_sql_wrapper(sql1, tb1.db_path, 10)
            vals1 = [row[0] for row in vals1]
            sql2 = f"SELECT DISTINCT `{col2_name}` FROM `{tb2.tb_name}`;"
            vals2 = execute_sql_wrapper(sql2, tb2.db_path, 10)
            vals2 = [row[0] for row in vals2]
            coverage = len(set(vals1) & set(vals2)) / min(
                len(set(vals1)), len(set(vals2))
            )
            if coverage > 0.95:
                col1_joinable_cols[col1_name].append((tb2.tb_name, col2_name))
                print(f"[{tb1.tb_name}].[{col1_name}] -> [{tb2.tb_name}].[{col2_name}]")
    return col1_joinable_cols


def find_potential_joinable_columns(db: Database) -> Dict:
    tbs = list(db.tables.keys())
    potential_joinable_columns = defaultdict(dict)
    for i, tb1 in enumerate(tbs):
        for j, tb2 in enumerate(tbs):
            if i >= j:
                continue
            joinable_cols = find_joinable_columns(db.tables[tb1], db.tables[tb2])
            if len(joinable_cols) > 0:
                potential_joinable_columns[tb1] = joinable_cols
    return potential_joinable_columns


if __name__ == "__main__":
    args = parse_option()
    qid_info = load_data(args.dataset_name)

    db_memory = Memory()
    flag = False
    for qid, info in qid_info.items():
        question = info["question"]
        evidence = info["evidence"]
        db_name = info["db_id"]
        if db_name in db_memory.memory:
            continue
        db = Database(args.dataset_name, db_name)
        db_memory.add(db_name, db)
        potential_joinable_columns = find_potential_joinable_columns(db)
        filename = f"../datasets/bird/dev/dev_databases/{db_name}/potential_joinable_columns.json"
        with open(filename, "w") as f:
            json.dump(potential_joinable_columns, f, indent=4)
        print(f"{filename} saved")
