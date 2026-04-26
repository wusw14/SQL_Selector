import json
from loader import load_preds
from loader import load_data
from utils import execute_sql_wrapper
from collections import defaultdict
import pandas as pd
import os
from llm_infer import llm_check
import argparse
import numpy as np


def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="birddev")
    parser.add_argument("--model_name", type=str, default="7B")
    parser.add_argument("--method_name", type=str, default="sql-r1")
    parser.add_argument("--align_column", action="store_true", default=False)
    parser.add_argument("--llm_check", action="store_true", default=False)
    parser.add_argument("--selector", type=str, default="majority")
    parser.add_argument("--save_eval", action="store_true", default=False)
    return parser.parse_args()


def load_minbug_qid_sqls(file_name):
    results = json.load(open(file_name))
    qid_sqls = {}
    qid_selected = {}
    for qid, res in results.items():
        sqls = res["sqls"]
        bugs = res["bugs"]
        if len(bugs) == 0:
            qid_selected[int(qid)] = "Error SQL"
            qid_sqls[int(qid)] = ["Error SQL"]
            continue
        min_bug = min(bugs)
        selected_sqls = []
        for sql, bug in zip(sqls, bugs):
            if bug == min_bug:
                selected_sqls.append(sql)
        if len(selected_sqls) == 0:
            qid_selected[int(qid)] = "Error SQL"
        else:
            qid_selected[int(qid)] = selected_sqls[0]
        qid_sqls[int(qid)] = selected_sqls
    return qid_sqls, qid_selected


def load_qid_sqls(file_name):
    results = json.load(open(file_name))
    qid_sqls = {}
    qid_selected = {}
    for qid, res in results.items():
        sql = res["selected_sql"]
        qid_selected[int(qid)] = sql
        sqls = []
        if "intent" in file_name:
            min_warning_cnt = None
            for item in res["sql_logs"]:
                if (
                    min_warning_cnt is None
                    or item["warning_cnt"] - item["exec_stats"]["columns"] * 0.01
                    < min_warning_cnt
                ):
                    min_warning_cnt = (
                        item["warning_cnt"] - item["exec_stats"]["columns"] * 0.01
                    )
            for item in res["sql_logs"]:
                if (
                    item["warning_cnt"] - item["exec_stats"]["columns"] * 0.01
                    == min_warning_cnt
                ):
                    sqls.append(item["sql"])
        else:
            for item in res["sql_logs"]:
                sqls.append(item["sql"])
        if len(sqls) == 0:
            qid_sqls[int(qid)] = [sql]
        else:
            qid_sqls[int(qid)] = sqls
    return qid_sqls, qid_selected


def align_column_with_gt(res, gt_res):
    if len(gt_res) == 0:
        return res
    if len(res) == 0 or len(res[0]) < len(gt_res[0]):
        return res
    if len(res[0]) == 1:
        return res
    # reorder the columns of res to match the columns of gt_res
    df_gt = pd.DataFrame(gt_res, columns=[f"col_{i}" for i in range(len(gt_res[0]))])
    df_res = pd.DataFrame(res, columns=[f"col_{i}" for i in range(len(res[0]))])
    column_order = []
    for i in range(len(gt_res[0])):
        # find the most similar column in df_res to the i-th column of df_gt
        most_similar_column = None
        most_similar_score = 0
        for j in range(len(res[0])):
            vals1 = df_gt.iloc[:, i].values
            vals2 = df_res.iloc[:, j].values
            score = len(set(vals1) & set(vals2))
            if score > most_similar_score and f"col_{j}" not in column_order:
                most_similar_score = score
                most_similar_column = j
        if most_similar_column is not None:
            column_order.append(f"col_{most_similar_column}")
    df_res = df_res[column_order]
    # transform df_res to tuples
    res = []
    for i in range(len(df_res)):
        res.append(tuple(df_res.iloc[i].values))
    res = tuple(res)
    return res


def finer_comparison(gt_res, gt_cols, res, res_col, evidence):
    if len(gt_res) != len(res):
        return False
    if len(gt_res) > 50:
        return False
    # let LLM check if res is the same as gt_res when ingoring the format difference
    prompt = f"""Task: Given the NL query, the execution result of the ground truth SQL and the execution result of the generated SQL, check if the generated SQL is correct for the NL query. You might refer to the evidence for understanding NL query if available. The column names of the execution results are also provided for reference.

NL Query: {question}

Evidence: {evidence}

Column Names of Ground Truth SQL Execution Result:
{gt_cols}
Ground Truth SQL Execution Result:
{gt_res}

Column Names of Generated SQL Execution Result:
{res_col}
Generated SQL Execution Result:
{res}

Respond with "Yes" if the execution results of the generated SQL is correct for the NL query by referring to the ground truth SQL execution result and the evidence, or "No" if it is incorrect. Provide no additional explanation.
"""
    if len(prompt) > 1000:
        return False
    res = llm_check([prompt])[0]
    # print(f"qid: {qid}, res: {res.strip()}")
    return res.startswith("Yes")


def flatten_tuple(res):
    flatten_res = []
    for sublist in res:
        for item in sublist:
            if item is not None:
                if type(item) == str:
                    flatten_res.append(item.lower())
                else:
                    flatten_res.append(item)
    return flatten_res


def eval_sql(gt_res_list, gt_cols_list, res, res_col, args, evidence):
    for gt_res in gt_res_list:
        if frozenset(res) == frozenset(gt_res):
            return 1, 1
    # align the columns of res to the columns of gt_res
    if len(gt_res_list) > 0 and args.align_column:
        for gt_res in gt_res_list:
            res_aligned = align_column_with_gt(res, gt_res)
            if frozenset(res_aligned) == frozenset(gt_res):
                return 1, 1
            # if flatten into one column and remove the null values, the result is the same as gt_res
            flatten_res = flatten_tuple(res)
            flatten_gt = flatten_tuple(gt_res)
            if frozenset(flatten_res) == frozenset(flatten_gt):
                return 1, 1

    # # llm check
    # for gt_res, gt_cols in zip(gt_res_list, gt_cols_list):
    #     # res_aligned = align_column_with_gt(res, gt_res)
    #     if args.llm_check and finer_comparison(gt_res, gt_cols, res, res_col, evidence):
    #         return 0, 1
    return 0, 0


if __name__ == "__main__":
    args = parse_argument()
    qid_info = load_data(args.dataset_name)
    print(f"len(qid_info): {len(qid_info)}")
    qid_unique_preds, qid_sql_cnt = load_preds(
        args.method_name, args.dataset_name, args.model_name
    )
    qid_sqls = qid_unique_preds
    print(f"len(qid_sqls): {len(qid_sqls)}")

    output_dir = (
        f"eval_results/{args.dataset_name}/{args.method_name}/{args.model_name}"
    )
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{args.selector}.json")

    # if os.path.exists(output_file):
    #     with open(output_file, "r") as f:
    #         eval_dict = json.load(f)
    # else:
    eval_dict = {}

    lower_acc1_list, lower_acc2_list = [], []
    upper_acc1_list, upper_acc2_list = [], []
    for qid, sqls in qid_sqls.items():
        if str(qid) in eval_dict:
            continue
        if qid not in qid_info:
            continue
        info = qid_info[qid]
        question = info["question"]
        evidence = info["evidence"]
        gt_sql = info["SQL"]
        db_id = info["db_id"]
        if args.dataset_name == "spidertest":
            db_path = f"../datasets/spider/test_database/{db_id}/{db_id}.sqlite"
        else:
            db_path = f"../datasets/bird/dev/dev_databases/{db_id}/{db_id}.sqlite"

        gt_res_list = []
        gt_cols_list = []
        if type(gt_sql) == list:
            for sql in gt_sql:
                res, cols = execute_sql_wrapper(
                    sql, db_path, 10, return_columns=True, normalized=True
                )
                if res not in [None, "Time Out", "Unexecutable"]:
                    gt_res_list.append(res)
                    gt_cols_list.append(cols)
        else:
            res, cols = execute_sql_wrapper(
                gt_sql, db_path, 10, return_columns=True, normalized=True
            )
            if res not in [None, "Time Out", "Unexecutable"]:
                gt_res_list.append(res)
                gt_cols_list.append(cols)

        # group sqls based on the execution result
        exec_map_dict = {}
        exec_res_sqls = defaultdict(list)
        exec_res_cols = defaultdict(list)
        for sql in sqls:
            res, cols = execute_sql_wrapper(
                sql, db_path, 10, return_columns=True, normalized=True
            )
            if res not in [None, "Time Out", "Unexecutable"]:
                exec_res_sqls[frozenset(res)].append(sql)
                exec_res_cols[frozenset(res)].append(cols)
                exec_map_dict[frozenset(res)] = res

        exec_res_acc = []
        lower_acc1, lower_acc2 = 1, 1
        upper_acc1, upper_acc2 = 0, 0
        for res, sqls in exec_res_sqls.items():
            res_raw = exec_map_dict[res]
            res_col = exec_res_cols[res][0]
            acc1, acc2 = eval_sql(
                gt_res_list,
                gt_cols_list,
                res_raw,
                res_col,
                args,
                evidence,
            )
            exec_res_acc.append({"sqls": sqls, "acc1": acc1, "acc2": acc2})
            lower_acc1 = min(lower_acc1, acc1)
            lower_acc2 = min(lower_acc2, acc2)
            upper_acc1 = max(upper_acc1, acc1)
            upper_acc2 = max(upper_acc2, acc2)
        lower_acc1 = min(lower_acc1, upper_acc1)
        lower_acc2 = min(lower_acc2, upper_acc2)
        lower_acc1_list.append(lower_acc1)
        lower_acc2_list.append(lower_acc2)
        upper_acc1_list.append(upper_acc1)
        upper_acc2_list.append(upper_acc2)
        eval_dict[qid] = exec_res_acc
        print(
            f"qid: {qid}, acc1: [{lower_acc1}, {upper_acc1}], acc2: [{lower_acc2}, {upper_acc2}]"
        )
        if args.save_eval:
            with open(output_file, "w") as f:
                json.dump(eval_dict, f, indent=2, ensure_ascii=False)
    print(
        f"Acc1: {np.mean(lower_acc1_list) * 100:.2f}%, {np.mean(upper_acc1_list) * 100:.2f}%"
    )
    print(
        f"Acc2: {np.mean(lower_acc2_list) * 100:.2f}%, {np.mean(upper_acc2_list) * 100:.2f}%"
    )
    print(f"Diff Cases: {sum(upper_acc1_list) - sum(lower_acc1_list)}")
