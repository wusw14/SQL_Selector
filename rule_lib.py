from typing import List, Dict
import numpy as np
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
from prompts import (
    get_rule_integration_prompt2,
    get_rule_relevance_prompt,
    get_rule_condition_prompt,
    get_rule_rank_prompt,
)
from llm_infer import llm_check
from utils import parse_json
import json
import argparse
from loader import load_all_preds, load_data
from representation import Representation
from sklearn.metrics import silhouette_score
from collections import defaultdict
import os
import re
import sqlparse
from sqlparse.tokens import Literal, Name, Number, String
import ast
import time
from copy import deepcopy


def abstract_sql(sql_string: str) -> str:
    """
    将输入的 SQL 语句进行骨架抽象化，移除具体的表名、列名和数值，仅保留结构关键字和运算符。

    例如:
    "SELECT * FROM cards WHERE color = 'B' AND age > 18"
    -> "SELECT * FROM IDENTIFIER WHERE IDENTIFIER = VALUE AND IDENTIFIER > VALUE"
    """
    # 1. 移除非必要的换行和多余空格，统一转大写
    sql_string = " ".join(sql_string.split())

    # 2. 利用 sqlparse 解析成 Token 列表
    parsed = sqlparse.parse(sql_string)
    if not parsed:
        return ""

    statement = parsed[0]
    abstracted_tokens = []

    # 3. 递归遍历所有 Token 并进行替换
    def flatten_and_replace(token_list):
        for token in token_list:
            # 如果是子列表（例如括号内的内容，嵌套查询等），递归处理
            if token.is_group:
                flatten_and_replace(token.tokens)
            else:
                ttype = token.ttype
                value = token.value.upper()

                # 情况 A: 字符串字面量 (如 'B', 'Artifact') -> 统一替换为 VALUE
                if ttype in String or ttype in Literal.String:
                    abstracted_tokens.append("VALUE")

                # 情况 B: 数字字面量 (如 18, 1984, 0.5) -> 统一替换为 VALUE
                elif ttype in Number or ttype in Literal.Number:
                    abstracted_tokens.append("VALUE")

                # 情况 C: 表名或列名 (如 cards, originaltype, id)
                # sqlparse 将它们归类为 Name 或 Name.Placeholder
                elif ttype in Name or str(ttype).startswith("Token.Name"):
                    abstracted_tokens.append("IDENTIFIER")

                # 情况 D: 普通关键字、运算符、标点符号 (如 SELECT, WHERE, =, >, AND) -> 原样保留
                else:
                    # 过滤掉空白 Token，防止干扰
                    if not token.is_whitespace:
                        abstracted_tokens.append(value)

    flatten_and_replace(statement.tokens)

    # 4. 重新拼接成规范的字符串
    result = " ".join(abstracted_tokens)

    # 5. 正则后处理补漏：处理某些被漏掉的带有包裹符号的列名/表名（如 `U-PRO`, "Patient"."ID"）
    # 将连续的 IDENTIFIER . IDENTIFIER 融合成一个，并将残余的引号路径抽象化
    result = re.sub(r"IDENTIFIER\s*\.\s*IDENTIFIER", "IDENTIFIER", result)
    result = re.sub(r'[`"\[\]\']\w+[`"\[\]\']', "IDENTIFIER", result)

    return result


def check_sql_matches_complex_condition(sqls: List[str], logic_cond: str) -> bool:
    """支持嵌套括号、特殊符号及自定义括号(如COUNT(DISTINCT))的 SQL 匹配检查器。"""
    # 1. 规范化 SQL 文本
    sql = " ".join(sqls)
    sql_upper = " ".join(sql.upper().split())

    # 2. 精准提取 Token
    # 使用正则将 &、|、(、) 作为分隔符，拆分出所有的基本条件文本
    # 注意：这里只拆分逻辑控制符，Token 内部自带的括号（如 COUNT(...)）通过更聪明的切分来保留
    raw_tokens = re.split(r"([&|()])", logic_cond)

    # 过滤掉逻辑运算符、括号以及纯空格，剩下的就是真正的 SQL 算子/表达式
    logic_symbols = {"&", "|", "(", ")", ""}
    tokens = set()
    for t in raw_tokens:
        t_clean = t.strip()
        if t_clean not in logic_symbols:
            tokens.add(t_clean)

    # 3. 检查每个 Token 是否在 SQL 中存在，并建立“安全变量名”的映射
    token_status = {}
    token_to_var = {}

    for idx, token in enumerate(tokens):
        token_upper = token.upper()

        # 判断 Token 是否存在于 SQL 中
        if token_upper.isalnum() or " " in token_upper or "(" in token_upper:
            # 针对包含空格、括号或纯文本的 Token，将其中的空格规范化为 \s+
            # 将其中的处理过的括号转义，确保匹配精准
            cleaned_pattern = re.escape(token_upper).replace(r"\ ", r"\s+")
            # 如果是纯单词，加边界符；如果带括号/符号，不加边界符（防 \b 报错）
            if token_upper.isalnum():
                pattern_str = rf"\b{cleaned_pattern}\b"
            else:
                pattern_str = cleaned_pattern

            token_status[token] = bool(re.search(pattern_str, sql_upper))
        else:
            # 纯符号（如 =, >, <）直接判断包含关系
            token_status[token] = token_upper in sql_upper

        # 生成合法的 Python 变量名
        token_to_var[token] = f"var_{idx}"

    # 4. 关键重构：安全替换
    # 按照长度从长到短排序 Token，防止子串拦截（比如先替换 COUNT(DISTINCT) 再替换 COUNT）
    sorted_tokens = sorted(tokens, key=len, reverse=True)

    py_expr = logic_cond
    for token in sorted_tokens:
        # 这里的替换必须非常小心，因为 token 内部可能带括号。
        # 我们使用 re.escape(token) 将其安全转化为正则字符串
        # 并且用 (<!\w) 和 (!\w) 代替 \b，防止带有括号的 token 边界匹配失效
        pattern = r"(?<![a-zA-Z0-9_])" + re.escape(token) + r"(?![a-zA-Z0-9_])"
        py_expr = re.sub(pattern, token_to_var[token], py_expr)

    # 替换逻辑运算符
    py_expr = py_expr.replace("&", " and ").replace("|", " or ")

    # 5. 通过构建安全 AST 并求值
    try:
        node = ast.parse(py_expr, mode="eval")

        def eval_node(n):
            if isinstance(n, ast.Expression):
                return eval_node(n.body)
            elif isinstance(n, ast.BoolOp):
                values = [eval_node(v) for v in n.values]
                if isinstance(n.op, ast.And):
                    return all(values)
                elif isinstance(n.op, ast.Or):
                    return any(values)
            elif isinstance(n, ast.Name):
                # 找到当前 var_x 对应的原始 token
                original_token = [k for k, v in token_to_var.items() if v == n.id][0]
                return token_status.get(original_token, False)
            else:
                raise ValueError(f"Unsupported syntax node: {type(n)}")

        return eval_node(node)

    except Exception as e:
        print(f"Error parsing logic_cond '{logic_cond}': {e}")
        print(f"Failed expression was: {py_expr}")
        return False


GENERAL_RULES = [
    "The SQL logic aligns with the NL query",
]


def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="birddev")
    parser.add_argument("--method_name", type=str, default="alphasql")
    parser.add_argument("--rule_mode", type=str, default="relevant")
    return parser.parse_args()


def get_skeleton(sqls: List[str]):
    sql_skeletons = []
    for sql in sqls:
        try:
            sql_skeleton = abstract_sql(sql)
            sql_skeletons.append(sql_skeleton)
        except:
            pass
    return sql_skeletons


class Rule:
    def __init__(
        self,
        text: str,
        sources: List[int],
        nl_cond=None,
        logic_cond=None,
        weight=0,
        pos_sql=None,
        neg_sql=None,
    ):
        self.text = text
        self.nl_cond = nl_cond
        self.sources = sources
        self.logic_cond = logic_cond
        self.weight = weight
        self.pos_sql = pos_sql
        self.neg_sql = neg_sql


class RuleCollection:
    def __init__(self, rules: List[Rule], qid_db: Dict[int, str]):
        self.rules = rules
        self.texts = [rule.text for rule in rules]
        self.nl_conds = [rule.nl_cond for rule in rules]
        self.logic_conds = [rule.logic_cond for rule in rules]
        self.weights = [rule.weight for rule in rules]
        self.emb_model = SentenceTransformer("BAAI/bge-base-en-v1.5", device="cuda")
        self.logic_cond_embs = self.encode(self.logic_conds)
        self.nl_cond_embs = self.encode(self.nl_conds)
        self.qids = self.get_qids()
        self.qid_db = qid_db
        self.db_qids = defaultdict(list)
        for qid, db in qid_db.items():
            self.db_qids[db].append(qid)

    def encode(self, conds: List[str]):
        cond_embs = self.emb_model.encode(conds, batch_size=128, device="cuda")
        cond_embs = cond_embs / np.linalg.norm(cond_embs, axis=1, keepdims=True)
        return cond_embs

    def get_qids(self):
        qids = []
        for rule in self.rules:
            qids.extend(rule.sources)
        qids = list(set(qids))
        return qids

    def update(self, rule):
        pass

    def retrieve(self, query_vector, top_k=5):
        sim_score = np.dot(query_vector, np.array(self.reps).T)
        top_k_indices = np.argsort(sim_score)[::-1]
        top_k_rules = [self.texts[i] for i in top_k_indices[:top_k]]
        return GENERAL_RULES + top_k_rules

    def retrieve_relevant(
        self, gp_sqls: List[str], qid, question, evidence, retrieval_mode, top_k=10
    ):
        """
        retrieve the relevant rules for the given query and gp_sqls
        :return: the texts and weghts of the relevant rules
        """
        if evidence is not None and len(evidence) > 0 and evidence.lower() != "none":
            general_rules = GENERAL_RULES[:1]
            weights = [1]
        else:
            general_rules = GENERAL_RULES
            weights = [1, 1]
        rule_score = {}
        rules = []
        rule_logic_cond_embs = []
        rule_nl_cond_embs = []
        for i, rule in enumerate(self.rules):
            if retrieval_mode == "crossdb":
                if len(set(rule.sources) - set(self.db_qids[self.qid_db[qid]])) == 0:
                    continue
            elif retrieval_mode == "crosscase":
                if len(set(rule.sources) - set([qid])) == 0:
                    continue
            rules.append(rule)
            rule_logic_cond_embs.append(self.logic_cond_embs[i])
            rule_nl_cond_embs.append(self.nl_cond_embs[i])
        # 1st: retrieve top-k rules by the embeddings
        start_time = time.time()
        gp_sql_skeletons = get_skeleton(gp_sqls)
        embs = self.encode([question])  #  + gp_sql_skeletons)
        question_emb = embs[:1]
        # gp_sqls_embs = embs[1:]
        nl_sim_score = np.dot(question_emb, np.array(rule_nl_cond_embs).T)[0]
        # logic_sim_scores = np.dot(gp_sqls_embs, np.array(rule_logic_cond_embs).T)
        # logic_sim_score = logic_sim_scores.max(axis=0)
        rule_score = nl_sim_score  # + logic_sim_score
        top_k_indices = np.argsort(rule_score)[::-1]
        top_k_rules = [rules[i] for i in top_k_indices[:top_k]]
        print(f"[1st step retrieval time]: {time.time() - start_time}")
        # 2nd: check if the logic_cond of rule is matched with the sqls
        start_time = time.time()
        rule_candidates = []
        for rule in rules:
            if check_sql_matches_complex_condition(gp_sqls, rule.logic_cond):
                rule_candidates.append(rule)
        if len(rule_candidates) == 0:
            return general_rules, weights
        print(f"[2nd step retrieval time]: {time.time() - start_time}")
        print(f"len(rule_candidates): {len(rule_candidates)}")
        rule_candidates = rule_candidates + top_k_rules
        rule_candidates = list(set(rule_candidates))
        org_size = len(rule_candidates)
        # # 3rd: rough filter by the LLM
        # start_time = time.time()
        # prompts = []
        # for rule in rule_candidates:
        #     prompt = get_rule_relevance_prompt(question, rule, gp_sqls)
        #     prompts.append(prompt)
        # responses = llm_check(prompts)
        # filtered_rule_candidates = []
        # for rule, response in zip(rule_candidates, responses):
        #     if response == "Yes":
        #         filtered_rule_candidates.append(rule)
        #         print(f"[DEBUG][rule]: {rule.text}")
        # print(f"[3rd step retrieval time]: {time.time() - start_time}")
        # print(f"len(rule_candidates): {len(filtered_rule_candidates)}")
        # rule_candidates = list(filtered_rule_candidates)
        # copy rule_candidates three times
        # divide the rules into batches, each batch size 20, and select top 5 for each batch
        start_time = time.time()
        for _ in range(1):
            indices_list = []
            prompts = []
            batch_rule_list = []
            indices = np.array(list(range(len(rule_candidates))))
            for t in range(2):
                np.random.shuffle(indices)
                # divide the indices into batches with a size of 25
                batch_size = 25
                for i in range(0, len(rule_candidates), batch_size):
                    indices_list.append(deepcopy(indices)[i : i + batch_size])
                    batch_rules = [rule_candidates[index] for index in indices_list[-1]]
                    batch_rule_list.append(batch_rules)
                    prompt = get_rule_rank_prompt(question, batch_rules, gp_sqls)
                    prompts.append(prompt)
            print(f"prompt size: {len(prompts)}")
            # let LLM check
            responses = llm_check(prompts)
            rule_cnt = defaultdict(int)
            for i, response in enumerate(responses):
                batch_rules = batch_rule_list[i]
                response = parse_json(response)
                print(response)
                indices = None
                if type(response) == dict and "hint_indices" in response:
                    indices = response["hint_indices"]
                elif "[" in response and "]" in response:
                    lindex = response.rindex("[")
                    rindex = response.rindex("]")
                    if rindex > lindex:
                        response = response[lindex + 1 : rindex]
                        response = response.replace('"', "").strip()
                if indices is not None:
                    for index in indices:
                        try:
                            rule_cnt[batch_rules[int(index)]] += 1
                        except:
                            pass
                else:
                    for rule in batch_rules:
                        rule_cnt[rule] += 1
            # select the rules with votes >= 3
            rule_candidates = []
            for rule, cnt in rule_cnt.items():
                if cnt >= 2:
                    rule_candidates.append(rule)
            if len(rule_candidates) <= 5:
                break
        print(f"[3rd step retrieval time]: {time.time() - start_time}")
        print(f"[Rule Size]: {org_size} -> {len(rule_candidates)}")
        rules = general_rules + rule_candidates
        weights = weights + [rule.weight for rule in rule_candidates]
        return rules, weights


def load_rule_collection(filename, qid_db, tgt_qids=None):
    data = json.load(open(filename, "r"))
    rules = []
    for item in data:
        if tgt_qids is not None and len(set(item["sources"]) & set(tgt_qids)) == 0:
            continue
        if "nl_cond" not in item or "logic_cond" not in item:
            continue
        if item["nl_cond"] is None or item["logic_cond"] is None:
            continue

        # text: str, source: int, nl_cond=None, logic_cond
        rule_obj = Rule(
            item["text"],
            item["sources"],
            item["nl_cond"] if "nl_cond" in item else None,
            item["logic_cond"] if "logic_cond" in item else None,
            item["weight"] if "weight" in item else 1,
        )
        rules.append(rule_obj)
    print(len(rules))
    rule_collection = RuleCollection(rules, qid_db)
    return rule_collection


def kmeans_by_optimizing_k(
    reps: List[np.ndarray], lower_bound: int, upper_bound: int, step: int = 1
) -> List[Rule]:
    # find the optimal k by optimizing the silhouette score
    upper_bound = min(len(reps) // 2, upper_bound)
    lower_bound = max(len(reps) // 50, lower_bound)
    best_k = lower_bound
    best_score = -1
    for k in range(lower_bound, upper_bound, step):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(reps)
        score = silhouette_score(reps, kmeans.labels_)
        if score > best_score:
            best_k = k
            best_score = score
        # print(k, score)
    # cluster the reps by the best k and return the labels of reps
    kmeans = KMeans(n_clusters=best_k, random_state=42)
    kmeans.fit(reps)
    print(f"[DEBUG][best_k]: {best_k}")
    print(f"[DEBUG][best_score]: {best_score}")
    return kmeans.labels_


def integrate_rule_by_llm(rules: List[Rule]) -> List[Rule]:
    if len(rules) == 1:
        return rules
    rule_texts = [rule.text for rule in rules]
    prompt = get_rule_integration_prompt2(rule_texts)
    print(f"[DEBUG][prompt]\n{prompt}\n--------------------------------\n")
    response = llm_check([prompt], llm="deepseek")[0]
    response = parse_json(response)
    integrated_rules = []
    for rule_indices, integrated_rule in response.items():
        print(rule_indices)
        print(integrated_rule)
        rule_indices = rule_indices.split(",")
        rule_indices = [int(index) - 1 for index in rule_indices]
        source_cases = []
        source_reps = []
        for index in rule_indices:
            source_cases.extend(rules[index].source_cases)
            source_reps.extend(rules[index].source_reps)
        integrated_rules.append(
            Rule(integrated_rule, "integrated", source_cases, source_reps)
        )
    return integrated_rules


def merge(rules: List[Rule], emb_model) -> List[Rule]:
    """
    clustering by the embeddings; and summarize rules for each cluster
    """
    integrated_rules = []
    # cluster the rules by the embeddings and then ask the LLM to integrate the rules in each cluster
    rule_texts = [rule.text for rule in rules]
    rule_embeddings = emb_model.encode(rule_texts, batch_size=128, device="cuda")
    rule_embeddings = rule_embeddings / np.linalg.norm(
        rule_embeddings, axis=1, keepdims=True
    )
    emb_labels = kmeans_by_optimizing_k(
        rule_embeddings, lower_bound=2, upper_bound=50, step=1
    )
    for sub_cluster_label in range(np.max(emb_labels) + 1):
        sub_cluster_rules = [
            rules[i] for i in range(len(rules)) if emb_labels[i] == sub_cluster_label
        ]
        sub_integrated_rules = integrate_rule_by_llm(sub_cluster_rules)
        integrated_rules.extend(sub_integrated_rules)
    return integrated_rules


def gen_hint_condition(rule: Rule, query: str, evidence: str, sqls: List[str]) -> Rule:
    rule_cond_prompt = get_rule_condition_prompt(rule.text, query, evidence, sqls)
    print(f"{rule_cond_prompt}\n\n")
    rule_cond_response = llm_check([rule_cond_prompt], llm="deepseek")[0]
    rule_cond_response = parse_json(rule_cond_response)
    rule.nl_cond = rule_cond_response["nl_condition"]
    rule.keywords = rule_cond_response["sql_keywords"]
    return rule


# if __name__ == "__main__":
#     database = "full_dev"
#     args = parse_option()
#     rule_file = f"results/{args.dataset_name}/alphasql/iterative_rules/{database}/post_process.json"
#     qid_preds, qid_sql_acc = load_all_preds(args)
#     qid_info = load_data(args.dataset_name)
#     """
#     results = json.load(open(rule_file, "r"))
#     qid_preds_with_rules = {}
#     qid_rules = defaultdict(list)
#     for qid, result in results.items():
#         if len(result["rules"]) == 0:
#             continue
#         rule_scores = result["rule_scores"]
#         for rule, score_dict in rule_scores.items():
#             pos_scores = score_dict["pos_scores"]
#             neg_scores = score_dict["neg_scores"]
#             pos_avg = np.mean(pos_scores)
#             neg_avg = np.mean(neg_scores)
#             if (
#                 score_dict["generality"] == 1
#                 and score_dict["clarity"] == 1
#                 and pos_avg > 0.9
#                 and pos_avg > neg_avg
#             ):
#                 qid_preds_with_rules[int(qid)] = qid_preds[int(qid)]
#                 qid_rules[int(qid)].append(rule)
#     representation_model = Representation(qid_preds_with_rules)
#     query_vectors = representation_model.query_vectors
#     rule_objs = []
#     for qid, rules in qid_rules.items():
#         for rule in rules:
#             rep = query_vectors[int(qid)]
#             rep = list(rep)
#             rule_obj = Rule(rule, "generated", [int(qid)], [rep])
#             rule_objs.append(rule_obj)
#     for t in range(2):
#         print(f"len(rule_objs): {len(rule_objs)}")
#         org_size = len(rule_objs)
#         rule_objs = merge(rule_objs, representation_model.emb_model)
#         new_size = len(rule_objs)
#         print(f"org_size: {org_size}, new_size: {new_size}")
#         rule_collection = RuleCollection(rule_objs)
#         # save the merged rules to the file
#         rules = []
#         for rule_obj in rule_objs:
#             rules.append(
#                 {
#                     "text": rule_obj.text,
#                     "mode": rule_obj.mode,
#                     "source_cases": rule_obj.source_cases,
#                     "source_reps": rule_obj.source_reps,
#                 }
#             )
#         output_file = f"results/birddev/alphasql/iterative_rules/{database}/merged_rules{t+1}.json"
#         with open(output_file, "w") as f:
#             json.dump(rules, f, indent=2)
#         if org_size - new_size <= 10:
#             print(f"Stop at t={t+1}")
#             break
#     """
#     rule_file = f"results/{args.dataset_name}/alphasql/iterative_rules/{database}/low_quality_rules.json"
#     low_quality_rules = json.load(open(rule_file, "r"))
#     rules = []
#     for rule_text, qid in low_quality_rules.items():
#         rule_obj = Rule(rule_text, "low_quality", [int(qid)], [])
#         rules.append(rule_obj)
#     rule_collection = RuleCollection(rules, {})
#     # rule_collection = load_rule_collection(rule_file)
#     output_file = f"results/{args.dataset_name}/alphasql/iterative_rules/{database}/low_quality_rule_with_cond.json"
#     if os.path.exists(output_file):
#         rule_with_cond = json.load(open(output_file, "r"))
#         processed_rules = [item["text"] for item in rule_with_cond]
#     else:
#         rule_with_cond = []
#     print(len(rule_with_cond))
#     for rule in rule_collection.rules:
#         if rule.text in processed_rules:
#             continue
#         info = qid_info[rule.source_cases[0]]
#         query = info["question"]
#         evidence = info["evidence"]
#         sqls = qid_preds[rule.source_cases[0]][:5]
#         rule_obj = gen_hint_condition(rule, query, evidence, sqls)
#         print(f"rule: {rule.text}")
#         print(f"nl_cond: {rule.nl_cond}")
#         print(f"keywords: {rule.keywords}")
#         print("-" * 100 + "\n" * 5)
#         rule_with_cond.append(
#             {
#                 "text": rule_obj.text,
#                 "nl_cond": rule_obj.nl_cond,
#                 "keywords": rule_obj.keywords,
#                 "mode": rule_obj.mode,
#                 "source_cases": rule_obj.source_cases,
#                 "source_reps": rule_obj.source_reps,
#             }
#         )
#         with open(output_file, "w") as f:
#             json.dump(rule_with_cond, f, indent=2)
