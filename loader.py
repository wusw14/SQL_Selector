import json
from collections import defaultdict


def load_data(dataset_name: str):
    qid_info = {}
    if dataset_name == "birddev":
        data = json.load(open(f"../datasets/bird/dev/dev_20251106.json", "r"))
        skipped_qids = []
        with open(f"../datasets/bird/dev/skip_qids", "r") as f:
            lines = f.readlines()
            skipped_qids = [int(line.strip()) for line in lines]
        for item in data:
            if item["question_id"] in skipped_qids:
                continue
            qid_info[item["question_id"]] = item
    elif dataset_name == "spider":
        data = json.load(open(f"../datasets/spider/test.json", "r"))
        for i, item in enumerate(data):
            qid_info[i] = {
                "question": item["question"],
                "db_id": item["db_id"],
                "SQL": item["query"],
                "evidence": None,
                "difficulty": item.get("difficulty", None),
            }
    else:
        raise ValueError(f"Invalid dataset name: {dataset_name}")
    return qid_info


def load_preds(method_name: str, dataset_name: str, model_name: str):
    """
    Return {"qid": [SQL1, SQL2, ...]}
    """
    if method_name == "alphasql":
        qid_preds = load_alphasql_preds(dataset_name, model_name)
    elif method_name == "sql-r1":
        qid_preds = load_sqlr1_preds(dataset_name, model_name)
    elif method_name == "artic_text2sql":
        qid_preds = load_artic_text2sql_preds(dataset_name, model_name)
    else:
        raise ValueError(f"Invalid method name: {method_name}")
    # TODO: deduplicate sqls
    qid_unique_preds = {}
    qid_sql_cnt = {}
    for qid, preds in qid_preds.items():
        qid_unique_preds[int(qid)] = list(set(preds))
        sql_cnt = defaultdict(int)
        for pred in preds:
            sql_cnt[pred] += 1
        qid_sql_cnt[int(qid)] = sql_cnt
    return qid_unique_preds, qid_sql_cnt


def load_alphasql_preds(dataset_name: str, model_name: str):
    """
    Return {"qid": [SQL1, SQL2, ...]}
    """
    filename = f"../Alpha-SQL/results/{dataset_name}/{model_name}_sqls.json"
    return json.load(open(filename, "r"))


def load_sqlr1_preds(dataset_name: str, model_name: str):
    """
    Return {"qid": [SQL1, SQL2, ...]}
    """
    filename = f"../SQL-R1/results/{dataset_name}-generated_sql_{model_name}.json"
    data = json.load(open(filename, "r"))
    qid_preds = {}
    for item in data:
        qid_preds[item["question_id"]] = item["pred_sqls"]
    return qid_preds


def load_artic_text2sql_preds(dataset_name: str, model_name: str):
    """
    Return {"qid": [SQL1, SQL2, ...]}
    """
    filename = f"../artic_text2sql/results/{dataset_name}.json"
    data = json.load(open(filename, "r"))
    qid_preds = {}
    for i, item in enumerate(data):
        qid_preds[i] = item["pred_sqls"]
    return qid_preds
