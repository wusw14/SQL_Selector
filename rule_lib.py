from typing import List
import numpy as np
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
from prompts import get_rule_integration_prompt2
from llm_infer import llm_check
from utils import parse_json
import json
import argparse
from loader import load_all_preds
from representation import Representation
from sklearn.metrics import silhouette_score


general_rules = [
    "The SQL must exactly reflect the intent of the natural language question.",
    "Should avoid adding assumptions not present in the input (including NL query, schema, evidence(optional)).",
    "When evidence defines metrics or calculations, SQL must prioritize and align with these definitions over general schema interpretation.",
]


def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="birddev")
    parser.add_argument("--method_name", type=str, default="alphasql")
    parser.add_argument("--rule_mode", type=str, default="relevant")
    return parser.parse_args()


class Rule:
    def __init__(self, text: str, mode: str, source_cases: List[int], source_reps):
        self.text = text
        self.mode = mode
        self.source_cases = source_cases
        self.source_reps = source_reps
        self.rep = np.mean(source_reps, axis=0).tolist()


class RuleCollection:
    def __init__(self, rules: List[Rule]):
        self.rules = rules
        self.reps = [rule.rep for rule in rules]
        self.texts = [rule.text for rule in rules]
        self.qids = self.get_qids()

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
        return general_rules + top_k_rules


def load_rule_collection(filename):
    data = json.load(open(filename, "r"))
    rules = []
    for item in data:
        if len(item["source_cases"]) < 3:
            continue
        rule_obj = Rule(
            item["text"], item["mode"], item["source_cases"], item["source_reps"]
        )
        rules.append(rule_obj)
    rule_collection = RuleCollection(rules)
    return rule_collection


def kmeans_by_optimizing_k(
    reps: List[np.ndarray], lower_bound: int, upper_bound: int, step: int = 1
) -> List[Rule]:
    # find the optimal k by optimizing the silhouette score
    upper_bound = min(len(reps) // 2, upper_bound)
    lower_bound = max(len(reps) // 20, lower_bound)
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


def merge_old(rules: List[Rule], emb_model) -> List[Rule]:
    """
    clustering by the reps; and summarize rules for each rep
    """
    reps = [rule.rep for rule in rules]
    rule_labels = kmeans_by_optimizing_k(reps, lower_bound=5, upper_bound=50, step=1)
    num_clusters = np.max(rule_labels) + 1
    print(f"[DEBUG][num_clusters]: {num_clusters}")
    integrated_rules = []
    for cluster_label in range(num_clusters):
        # deduplicate the rules in each cluster
        cluster_rules = [
            rules[i] for i in range(len(rules)) if rule_labels[i] == cluster_label
        ]
        if len(cluster_rules) <= 20:
            sub_integrated_rules = integrate_rule_by_llm(cluster_rules)
            integrated_rules.extend(sub_integrated_rules)
        else:
            # cluster the rules by the embeddings and then ask the LLM to integrate the rules in each cluster
            rule_texts = [rule.text for rule in cluster_rules]
            rule_embeddings = emb_model.encode(
                rule_texts, batch_size=128, device="cuda"
            )
            rule_embeddings = rule_embeddings / np.linalg.norm(
                rule_embeddings, axis=1, keepdims=True
            )
            emb_labels = kmeans_by_optimizing_k(
                rule_embeddings, lower_bound=2, upper_bound=20, step=1
            )
            for sub_cluster_label in range(np.max(emb_labels) + 1):
                sub_cluster_rules = [
                    cluster_rules[i]
                    for i in range(len(cluster_rules))
                    if emb_labels[i] == sub_cluster_label
                ]
                sub_integrated_rules = integrate_rule_by_llm(sub_cluster_rules)
                integrated_rules.extend(sub_integrated_rules)
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


if __name__ == "__main__":
    rule_file = "results/birddev/alphasql/iterative_rules/full_dev/t=4.json"
    args = parse_option()
    qid_preds, qid_sql_acc = load_all_preds(args)
    results = json.load(open(rule_file, "r"))
    qid_preds_with_rules = {}
    for qid, result in results.items():
        rules = result["rules"]
        if len(rules) == 0:
            continue
        qid_preds_with_rules[int(qid)] = qid_preds[int(qid)]
    representation_model = Representation(qid_preds_with_rules)
    query_vectors = representation_model.query_vectors
    rule_objs = []
    for qid, result in results.items():
        rules = result["rules"]
        if len(rules) == 0:
            continue
        for rule in rules[-1]:
            rep = query_vectors[int(qid)]
            rep = list(rep)
            rule_obj = Rule(rule, "generated", [int(qid)], [rep])
            rule_objs.append(rule_obj)
    #         print(rep, type(rep))
    # exit()
    print(f"len(rule_objs): {len(rule_objs)}")

    rule_objs = merge(rule_objs, representation_model.emb_model)
    rule_collection = RuleCollection(rule_objs)
    # save the merged rules to the file
    rules = []
    for rule_obj in rule_objs:
        rules.append(
            {
                "text": rule_obj.text,
                "mode": rule_obj.mode,
                "source_cases": rule_obj.source_cases,
                "source_reps": rule_obj.source_reps,
            }
        )
    output_file = "results/birddev/alphasql/iterative_rules/full_dev/merged_rules.json"
    with open(output_file, "w") as f:
        json.dump(rules, f, indent=2)
