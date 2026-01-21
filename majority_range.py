import json
from loader import load_preds
from loader import load_data
from utils import execute_sql_wrapper
from collections import defaultdict


def load_qid_sqls(file_name):
    if "verifier" in file_name:
        sqls_raw = json.load(open(file_name))
        qid_sqls = {}
        for qid, res in sqls_raw.items():
            sqls = res["pred_sqls"]
            correctness_list = res["correctness_list"]
            if len(sqls) == 0:
                continue
            selected_sqls = []
            for sql, correctness in zip(sqls, correctness_list):
                if correctness == "Yes":
                    selected_sqls.append(sql)
            if len(selected_sqls) == 0:
                qid_sqls[int(qid)] = list(set(sqls))
            else:
                qid_sqls[int(qid)] = list(set(selected_sqls))
        return qid_sqls
    else:
        sqls_raw = json.load(open(file_name))
        qid_sqls = {}
        for qid, res in sqls_raw.items():
            sqls = res["sqls"]
            bugs = res["bugs"]
            if len(sqls) == 0:
                continue
            min_bug = min(bugs)
            selected_sqls = []
            for sql, bug in zip(sqls, bugs):
                if bug == min_bug:
                    selected_sqls.append(sql)
            qid_sqls[int(qid)] = list(set(selected_sqls))
    return qid_sqls


if __name__ == "__main__":
    # file_name = "/ssddata/wushw/text2sql/SQL-R1/results/birddev-sql_7B_all.json"
    file_name = "/ssddata/wushw/text2sql/verifier/results/Qwen3/birddev/parser/sql-r1_7B_probs.json"
    qid_sqls = load_qid_sqls(file_name)
    print(f"len(qid_sqls): {len(qid_sqls)}")
    qid_info = load_data("birddev")
    _, qid_sql_cnt = load_preds("sql-r1", "birddev", "7B")
    lower_acc_sum, upper_acc_sum = 0, 0
    for qid, sqls in qid_sqls.items():
        info = qid_info[qid]
        question = info["question"]
        evidence = info["evidence"]
        gt_sql = info["SQL"]
        db_id = info["db_id"]
        db_path = f"/ssddata/wushw/text2sql/datasets/bird/dev/dev_databases/{db_id}/{db_id}.sqlite"
        gt_res = execute_sql_wrapper(gt_sql, db_path, 10)
        res_cnt = defaultdict(int)
        for sql in sqls:
            res = execute_sql_wrapper(sql, db_path, 10)
            res_cnt[frozenset(res)] += qid_sql_cnt[qid][sql]
        max_cnt = max(res_cnt.values())
        lower_acc, upper_acc = 1, 0
        for res, cnt in res_cnt.items():
            if cnt == max_cnt:
                if res == frozenset(gt_res):
                    upper_acc = 1
                else:
                    lower_acc = 0
        lower_acc = min(lower_acc, upper_acc)
        print(f"qid: {qid}, lower_acc: {lower_acc}, upper_acc: {upper_acc}")

        lower_acc_sum += lower_acc
        upper_acc_sum += upper_acc
    lower_acc_avg = lower_acc_sum / len(qid_info) * 100
    upper_acc_avg = upper_acc_sum / len(qid_info) * 100
    print(f"lower_acc_avg: {lower_acc_avg:.2f}%, upper_acc_avg: {upper_acc_avg:.2f}%")
