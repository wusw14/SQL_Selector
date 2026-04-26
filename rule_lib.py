from typing import List, Dict
import numpy as np
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
from prompts import (
    get_rule_integration_prompt2,
    get_rule_relevance_prompt,
    get_rule_condition_prompt,
)
from llm_infer import llm_check
from utils import parse_json
import json
import argparse
from loader import load_all_preds, load_data
from representation import Representation
from sklearn.metrics import silhouette_score
from collections import defaultdict


GENERAL_RULES = [
    "Follow the instructions in the evidence, particularly regarding return values and metric calculations.",
]

SQL_KEYWORDS = [
    "LIMIT",
    "LIMIT 1",
    "COUNT",
    "AVG",
    "MIN",
    "MAX",
    "SUM",
    "DISTINCT",
    "NULL",
    "EXISTS",
    "LIKE",
    "CASE",
    "PARTITION BY",
    "ORDER BY",
    "GROUP BY",
    "CAST",
    "RANK",
    "ROW_NUMBER",
]


def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="birddev")
    parser.add_argument("--method_name", type=str, default="alphasql")
    parser.add_argument("--rule_mode", type=str, default="relevant")
    return parser.parse_args()


class Rule:
    def __init__(
        self,
        text: str,
        mode: str,
        source_cases: List[int],
        source_reps,
        nl_cond=None,
        keywords=[],
    ):
        self.text = text
        self.mode = mode
        self.source_cases = source_cases
        self.source_reps = source_reps
        self.rep = np.mean(source_reps, axis=0).tolist()
        self.nl_cond = nl_cond
        self.keywords = self.update_keywords(keywords)

    def update_keywords(self, keywords):
        filtered_keywords = []
        for k in keywords:
            if k.upper() in SQL_KEYWORDS:
                filtered_keywords.append(k.upper())
        return filtered_keywords


class RuleCollection:
    def __init__(self, rules: List[Rule], qid_db: Dict[int, str]):
        self.rules = rules
        self.reps = [rule.rep for rule in rules]
        self.texts = [rule.text for rule in rules]
        self.qids = self.get_qids()
        self.qid_db = qid_db
        self.db_qids = defaultdict(list)
        for qid, db in qid_db.items():
            self.db_qids[db].append(qid)

    def get_qids(self):
        qids = []
        for rule in self.rules:
            qids.extend(rule.source_cases)
        qids = list(set(qids))
        return qids

    def update(self, rule):
        pass

    def retrieve(self, query_vector, top_k=5):
        sim_score = np.dot(query_vector, np.array(self.reps).T)
        top_k_indices = np.argsort(sim_score)[::-1]
        top_k_rules = [self.texts[i] for i in top_k_indices[:top_k]]
        return GENERAL_RULES + top_k_rules

    def retrieve_relevant(self, sqls: List[str], qid, question):
        prompts, org_rules = [], []
        rule_score = {}
        rule_hitted_keywords = {}
        for rule in self.rules:
            same_db_qids = self.db_qids[self.qid_db[qid]]
            diff_db_qids = list(set(rule.source_cases) - set(same_db_qids))
            if len(diff_db_qids) == 0:
                continue
            related_cnt = 0
            hitted_keywords = set()
            for sql in sqls:
                flag = False
                for keyword in rule.keywords:
                    if keyword in sql.upper():
                        hitted_keywords.add(keyword)
                        flag = True
                related_cnt += int(flag)
            rule_score[rule] = related_cnt
            rule_hitted_keywords[rule] = hitted_keywords
        # sort the rules by the score
        rule_score = sorted(rule_score.items(), key=lambda x: x[1], reverse=True)
        keywords_set = set()
        for rule, score in rule_score:
            keyword_uncovred_cnt = len(set(rule_hitted_keywords[rule]) - keywords_set)
            if score == len(sqls) or (
                score > 0 and (len(org_rules) < 20 or keyword_uncovred_cnt > 0)
            ):
                prompt = get_rule_relevance_prompt(question, rule.nl_cond, sqls)
                prompts.append(prompt)
                org_rules.append(rule.text)
                keywords_set.update(set(rule_hitted_keywords[rule]))
        if len(prompts) == 0:
            return GENERAL_RULES

        responses = llm_check(prompts)
        rules = []
        for rule, response in zip(org_rules, responses):
            if response == "Yes":
                rules.append(rule)
            # print(f"[DEBUG][rule]: {rule}, relevance: {response}")
        print(f"[Rule Size]: {len(org_rules)} -> {len(rules)}")
        return GENERAL_RULES + rules


def load_rule_collection(filename, qid_db):
    data = json.load(open(filename, "r"))
    rules = []
    for item in data:
        rule_obj = Rule(
            item["text"],
            item["mode"],
            item["source_cases"],
            item["source_reps"],
            item["nl_cond"],
            item["keywords"],
        )
        rules.append(rule_obj)
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
#     rule_file = f"results/birddev/alphasql/iterative_rules/{database}/post_process.json"
#     args = parse_option()
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
# rule_file = (
#     f"results/birddev/alphasql/iterative_rules/{database}/low_quality_rules.json"
# )
# low_quality_rules = json.load(open(rule_file, "r"))
# rules = []
# for rule_text, qid in low_quality_rules.items():
#     rule_obj = Rule(rule_text, "low_quality", [int(qid)], [])
#     rules.append(rule_obj)
# rule_collection = RuleCollection(rules, {})
# # rule_collection = load_rule_collection(rule_file)
# rule_with_cond = []
# for rule in rule_collection.rules:
#     info = qid_info[rule.source_cases[0]]
#     query = info["question"]
#     evidence = info["evidence"]
#     sqls = qid_preds[rule.source_cases[0]][:5]
#     rule_obj = gen_hint_condition(rule, query, evidence, sqls)
#     print(f"rule: {rule.text}")
#     print(f"nl_cond: {rule.nl_cond}")
#     print(f"keywords: {rule.keywords}")
#     print("-" * 100 + "\n" * 5)
#     rule_with_cond.append(
#         {
#             "text": rule_obj.text,
#             "nl_cond": rule_obj.nl_cond,
#             "keywords": rule_obj.keywords,
#             "mode": rule_obj.mode,
#             "source_cases": rule_obj.source_cases,
#             "source_reps": rule_obj.source_reps,
#         }
#     )
#     output_file = f"results/birddev/alphasql/iterative_rules/{database}/low_quality_rule_with_cond.json"
#     with open(output_file, "w") as f:
#         json.dump(rule_with_cond, f, indent=2)
