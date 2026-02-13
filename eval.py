import json
from loader import load_preds
from loader import load_data
from utils import execute_sql_wrapper
from collections import defaultdict
import pandas as pd
import os
from llm_infer import llm_check
import argparse


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
        qid_selected[int(qid)] = None
        qid_sqls[int(qid)] = selected_sqls
    return qid_sqls, qid_selected


def load_qid_sqls(file_name):
    if not os.path.exists(file_name):
        print(f"File {file_name} does not exist")
        exit()
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


def refine_selection(file_name, args):
    results = json.load(open(file_name))
    gp_sql_acc_file = f"eval_results/{args.dataset_name}/{args.method_name}/{args.model_name}/gp_sql_acc.json"
    gp_sql_acc = json.load(open(gp_sql_acc_file, "r"))
    qid_sqls = {}
    qid_selected = {}
    for qid, res in results.items():
        if str(qid) not in gp_sql_acc:
            continue
        if "comparison_notes" not in res:
            qid_selected[int(qid)] = res["selected_sql"]
            qid_sqls[int(qid)] = [res["selected_sql"]]
            continue
        comparison_notes = res["comparison_notes"]
        pair_sql_vote = {}
        for note in comparison_notes:
            sql1 = note["SQL1"]
            sql2 = note["SQL2"]
            if type(note["comparison_note"]) == str:
                better_sql = "Unsure"
            else:
                better_sql = note["comparison_note"]["better_sql"]
            if better_sql == "SQL1":
                pair_sql_vote[(sql1, sql2)] = 1
                pair_sql_vote[(sql2, sql1)] = -1
            elif better_sql == "SQL2":
                pair_sql_vote[(sql1, sql2)] = -1
                pair_sql_vote[(sql2, sql1)] = 1
            else:
                pair_sql_vote[(sql1, sql2)] = 0
                pair_sql_vote[(sql2, sql1)] = 0
        sqls = []
        for item in res["sql_logs"]:
            sqls.append(item["sql"])
        gp_sqls_list = gp_sql_acc.get(str(qid), [])
        sql_per_gp = []
        for gp_sqls in gp_sqls_list:
            # compare the sql within each group
            sql_votes = defaultdict(int)
            for sql in gp_sqls["sqls"]:
                if sql in sqls:
                    for sql1, sql2 in pair_sql_vote:
                        if sql1 == sql and sql2 in sqls:
                            sql_votes[sql1] += pair_sql_vote[(sql1, sql2)]
            sorted_sql_votes = sorted(
                sql_votes.items(), key=lambda x: x[1], reverse=True
            )
            if len(sorted_sql_votes) == 0:
                continue
            sql_per_gp.append(sorted_sql_votes[0][0])
        # compare the sql across different groups
        while len(sql_per_gp) > 1:
            sql_votes = defaultdict(int)
            max_vote = -1
            for sql1 in sql_per_gp:
                for sql2 in sql_per_gp:
                    if sql1 == sql2:
                        continue
                    sql_votes[sql1] += pair_sql_vote.get((sql1, sql2), 0)
                max_vote = max(max_vote, sql_votes[sql1])
            min_vote = min(sql_votes.values())
            sql_per_gp = [sql for sql in sql_per_gp if sql_votes[sql] == max_vote]
            if max_vote == min_vote:
                break
        selected_sql = sql_per_gp[0]
        qid_selected[int(qid)] = selected_sql
        qid_sqls[int(qid)] = sqls
    return qid_sqls, qid_selected


def load_selected(args):
    if args.selector == "majority":
        qid_unique_preds, qid_sql_cnt = load_preds(
            args.method_name, args.dataset_name, args.model_name
        )
        qid_sqls = qid_unique_preds
        qid_selected = {qid: None for qid in qid_sqls.keys()}
    elif args.selector == "minbug":
        if args.method_name == "sql-r1":
            file_name = f"../SQL-R1/results/{args.dataset_name}-generated_sql_{args.model_name}_bugchecked.json"
        elif args.method_name == "alphasql":
            # TODO: change the file name
            file_name = f"../Alpha-SQL/results/{args.dataset_name}/{args.model_name}_sqls_bugchecked.json"
        elif args.method_name == "artic_text2sql":
            # TODO: change the file name
            file_name = f"../artic_text2sql/results/{args.dataset_name}.json"
        else:
            raise ValueError(f"Invalid method name: {args.method_name}")
        qid_sqls, qid_selected = load_minbug_qid_sqls(file_name)
    elif (
        args.selector in ["exec", "table", "join"]
        or "intent" in args.selector
        or "pairwise" in args.selector
        or "our" in args.selector
    ):
        file_name = f"results/{args.dataset_name}/{args.method_name}/{args.model_name}/{args.selector}.json"
        qid_sqls, qid_selected = load_qid_sqls(file_name)
    elif "exhaustive" in args.selector:
        file_name = f"results/{args.dataset_name}/{args.method_name}/{args.model_name}/{args.selector}.json"
        qid_sqls, qid_selected = refine_selection(file_name, args)
    else:
        raise ValueError(f"Invalid selector: {args.selector}")
    return qid_sqls, qid_selected


if __name__ == "__main__":
    args = parse_argument()
    qid_info = load_data(args.dataset_name)
    qid_unique_preds, qid_sql_cnt = load_preds(
        args.method_name, args.dataset_name, args.model_name
    )

    qid_sqls, qid_selected = load_selected(args)
    # print(f"========={args.selector}========")
    # print(f"len(qid_sqls): {len(qid_sqls)}")
    # print(f"len(qid_selected): {len(qid_selected)}")
    # exit()

    output_dir = (
        f"eval_results/{args.dataset_name}/{args.method_name}/{args.model_name}"
    )
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{args.selector}.json")

    gp_sql_acc_file = f"{output_dir}/gp_sql_acc.json"
    qid_gp_sql_acc = json.load(open(gp_sql_acc_file, "r"))

    lower_acc_sum, upper_acc_sum = 0, 0
    maj_lower_acc_sum, maj_upper_acc_sum = 0, 0
    selected_acc_sum = 0
    db_cnt = defaultdict(int)
    # if os.path.exists(output_file):
    #     with open(output_file, "r") as f:
    #         eval_dict = json.load(f)
    # else:
    eval_dict = {}

    for qid, sqls in qid_sqls.items():
        if str(qid) in eval_dict:
            continue
        if qid not in qid_info:
            continue
        if len(sqls) == 0:
            continue
        selected_sql = qid_selected.get(qid, sqls[0])
        sql_cnt = qid_sql_cnt.get(qid, {})
        info = qid_info[qid]
        question = info["question"]
        evidence = info["evidence"]
        gt_sql = info["SQL"]
        db_id = info["db_id"]
        db_path = f"../datasets/bird/dev/dev_databases/{db_id}/{db_id}.sqlite"
        gp_sql_acc = qid_gp_sql_acc.get(str(qid), [])
        gp_votes = {}
        gp_acc = {}
        max_vote = 0
        selected_acc = 0
        checked_sql = []
        for gp, sql_acc in enumerate(gp_sql_acc):
            gp_sqls = sql_acc["sqls"]
            if args.llm_check:
                acc = sql_acc["acc2"]
            else:
                acc = sql_acc["acc1"]
            vote = 0
            for sql in gp_sqls:
                if sql in sqls:
                    vote += sql_cnt.get(sql, 1)
                    checked_sql.append(sql)
                if sql == selected_sql:
                    selected_acc = acc
            if vote > 0:
                gp_votes[gp] = vote
                gp_acc[gp] = acc
                max_vote = max(max_vote, vote)
        new_gp = len(gp_acc) + 1
        vote = 0
        for sql in sqls:
            if sql not in checked_sql:
                vote += sql_cnt.get(sql, 1)
                checked_sql.append(sql)
                gp_acc[new_gp] = 0
                gp_votes[new_gp] = vote
        max_vote = max(max_vote, vote)
        if len(gp_acc) == 0:
            continue
        lower_acc = min(gp_acc.values())
        upper_acc = max(gp_acc.values())
        maj_lower_acc, maj_upper_acc = 1, 0
        max_vote_cnt = 0
        for gp, vote in gp_votes.items():
            if vote == max_vote:
                maj_lower_acc = min(maj_lower_acc, gp_acc[gp])
                maj_upper_acc = max(maj_upper_acc, gp_acc[gp])
                max_vote_cnt += 1
        if "our" not in args.selector:
            if max_vote_cnt > 1:
                selected_acc = maj_upper_acc / max_vote_cnt
            else:
                selected_acc = maj_upper_acc
        eval_dict[qid] = {
            "upper_acc": upper_acc,
            "lower_acc": lower_acc,
            "maj_lower_acc": maj_lower_acc,
            "maj_upper_acc": maj_upper_acc,
            "selected_acc": selected_acc,
        }
        # if selected_acc != upper_acc:
        #     print(
        #         f"qid: {qid}, upper_acc: {upper_acc}, lower_acc: {lower_acc}, maj_lower_acc: {maj_lower_acc}, maj_upper_acc: {maj_upper_acc}, selected_acc: {selected_acc}"
        #     )
        if args.save_eval:
            with open(output_file, "w") as f:
                json.dump(eval_dict, f, indent=2, ensure_ascii=False)

    eval_base = json.load(open(os.path.join(output_dir, "majority.json"), "r"))
    for qid in qid_info:
        if qid not in eval_dict and str(qid) not in eval_dict:
            try:
                eval_dict[qid] = eval_base[str(qid)]
            except Exception as e:
                # print(f"Error: {e}")
                eval_dict[qid] = {
                    "upper_acc": 0,
                    "lower_acc": 0,
                    "maj_lower_acc": 0,
                    "maj_upper_acc": 0,
                    "selected_acc": 0,
                }

    # print(f"len(eval_dict): {len(eval_dict)}")
    for qid, res in eval_dict.items():
        upper_acc_sum += res["upper_acc"]
        lower_acc_sum += min(res["lower_acc"], res["upper_acc"])
        maj_lower_acc_sum += min(res["maj_lower_acc"], res["maj_upper_acc"])
        maj_upper_acc_sum += res["maj_upper_acc"]
        selected_acc_sum += res["selected_acc"]
    upper_acc_ratio = upper_acc_sum / len(qid_info) * 100
    lower_acc_ratio = lower_acc_sum / len(qid_info) * 100
    maj_lower_acc_ratio = maj_lower_acc_sum / len(qid_info) * 100
    maj_upper_acc_ratio = maj_upper_acc_sum / len(qid_info) * 100
    selected_acc_ratio = selected_acc_sum / len(qid_info) * 100
    method_name = f"{args.method_name}({args.model_name})"
    output_list = [f"{method_name:^30}"]
    output_list.append(f"{args.selector:^10}")
    output_list.append(f"{lower_acc_ratio:.2f}%")
    output_list.append(f"{upper_acc_ratio:.2f}%")
    output_list.append(f"{maj_lower_acc_ratio:.2f}%")
    output_list.append(f"{maj_upper_acc_ratio:.2f}%")
    output_list.append(f"{selected_acc_ratio:.2f}%")
    print(" | ".join(output_list))
