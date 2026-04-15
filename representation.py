import json
from collections import defaultdict
from typing import List, Dict
from loader import load_preds
import numpy as np
from sentence_transformers import SentenceTransformer
import random
from copy import deepcopy

general_rules = [
    "The SQL must exactly reflect the intent of the natural language question.",
    "Avoid adding assumptions not present in the input (including NL query, schema, evidence(optional)).",
    "If one SQL is semantically wrong (e.g., missing or redundant filters, wrong joins, wrong columns), reject it.",
]

keywords = [
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


def count_n_grams(text: str, n: int) -> dict:
    text = text.upper()
    n_grams = defaultdict(int)
    splitted_text = text.split()
    for i in range(len(splitted_text) - n + 1):
        n_gram = " ".join(splitted_text[i : i + n])
        n_grams[n_gram] += 1
    return n_grams


def count_n_grams_in_SQLs(sqls: List[str], n: int) -> dict:
    n_grams = defaultdict(int)
    for sql in sqls:
        for k, v in count_n_grams(sql, n).items():
            n_grams[k] += v
    return n_grams


def get_representation(sqls: str) -> dict:
    dimensions = [0] * len(keywords)
    for sql in sqls:
        for i, keyword in enumerate(keywords):
            if keyword in sql.upper():
                dimensions[i] += 1
    return dimensions


def get_idf(dimensions: List[int]) -> List[float]:
    dimensions = np.array(dimensions)
    df = np.sum(dimensions > 0, axis=0)
    idf = np.log((len(dimensions) + 1) / (df + 1)) + 1.0
    return idf


class Representation:
    def __init__(self, query_sqls: Dict[str, List[str]]):
        self.query_vectors = self.build_query_vectors(query_sqls)
        self.emb_model = SentenceTransformer("BAAI/bge-base-en-v1.5", device="cuda")

    def norm(self, dimension):
        dimension = np.array(dimension)
        dimension = dimension * self.idf
        return dimension

    def build_query_vectors(
        self, query_sqls: Dict[str, List[str]]
    ) -> Dict[str, List[float]]:
        query_vectors = {}
        q_df_list = []
        for qid, sqls in query_sqls.items():
            q_df = get_representation(sqls)
            q_df_list.append(q_df)
            query_vectors[qid] = q_df
        self.idf = get_idf(q_df_list)
        for qid, q_df in query_vectors.items():
            query_vectors[qid] = self.norm(q_df)
        print(f"len(query_vectors): {len(query_vectors)}")
        return query_vectors

    def get_top_similar_queries(
        self, sqls: List[str], top_k: int, exclude_qids: List[int] = None
    ) -> List[str]:
        vector = get_representation(sqls)
        vector = self.norm(vector)
        qids, vectors = zip(*self.query_vectors.items())
        similarities = np.dot(vectors, vector)
        top_k_indices = np.argsort(similarities)[::-1]
        rel_qids = []
        for index in top_k_indices:
            if exclude_qids is None or qids[index] not in exclude_qids:
                rel_qids.append(qids[index])
                if len(rel_qids) >= top_k:
                    break
        return rel_qids


# get_relevant_rules(qid_rules, preds, representation_model, top_k=10)
def get_relevant_rules(
    qid: int,
    qid_rules: Dict[int, Dict],
    sqls: List[str],
    representation_model: Representation,
    rule_mode: str,
    top_k: int,
    exclude_qids: List[int] = None,
) -> List[str]:
    if qid in qid_rules and rule_mode == "ideal":
        return qid_rules[qid]
    if rule_mode == "random":
        relevant_qids = random.sample(list(qid_rules.keys()), top_k)
    else:
        relevant_qids = representation_model.get_top_similar_queries(
            sqls, top_k, exclude_qids=exclude_qids
        )
    collected_rules = []
    for qid in relevant_qids:
        collected_rules.extend(qid_rules[qid])
    # get the embeddings of the collected rules
    rule_embeddings = representation_model.emb_model.encode(
        collected_rules, batch_size=128, device="cuda"
    )
    # normalize the rule embeddings
    rule_embeddings = rule_embeddings / np.linalg.norm(
        rule_embeddings, axis=1, keepdims=True
    )
    rule_embedding_dict = {
        rule: embedding for rule, embedding in zip(collected_rules, rule_embeddings)
    }
    selected_rules = deepcopy(qid_rules[relevant_qids[0]])
    selected_embs = [rule_embedding_dict[rule] for rule in selected_rules]
    for qid in relevant_qids[1:]:
        rules = qid_rules[qid]
        for rule in rules:
            if len(selected_rules) >= 3:
                break
            embedding = rule_embedding_dict[rule]
            similarity = np.dot(embedding, np.array(selected_embs).T)
            if max(similarity) < 0.85:
                selected_rules.append(rule)
                selected_embs.append(embedding)
    return general_rules + selected_rules


# if __name__ == "__main__":
#     qid_pred, qid_sql_cnt = load_preds("alphasql", "birddev", "Qwen2.5-7B")
#     sqls_all = []
#     for qid, sqls in qid_pred.items():
#         sqls_all.extend(sqls)
#     n_grams = count_n_grams_in_SQLs(sqls_all, 2)
#     n_grams = sorted(n_grams.items(), key=lambda x: x[1], reverse=True)
#     for n_gram, cnt in n_grams[:100]:
#         print(f"{n_gram}: {cnt}")
