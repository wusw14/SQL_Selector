from parser import SQLCollection, SQLNode
from typing import List, Tuple
import pandas as pd
import numpy as np
import json
from syntax import extract_group_by, classify_ranking
import re


def order_vs_maxmin(obj1, obj2):
    note = f"{obj1} uses `ORDER BY` with `LIMIT`, while {obj2} uses `MAX`/`MIN`."
    note += "\nHint 1: Pay attention! Verify if the `ORDER BY` object needs to be converted to the `REAL` type."
    note += "\nHint 2: In general, the `MAX`/`MIN` function is preferred as it handles tied scenarios more robustly."
    return note


def same_returned_columns(
    sql_node1: SQLNode, sql_node2: SQLNode
) -> Tuple[List[str], List[str]]:
    sql1_cols = sql_node1.exec_columns
    sql2_cols = sql_node2.exec_columns
    # check if each column in sql_node1 could be found in sql_node2
    df_sql1 = pd.DataFrame(sql_node1.exec_res, columns=sql1_cols)
    df_sql2 = pd.DataFrame(sql_node2.exec_res, columns=sql2_cols)
    sim_matrix = np.zeros((len(sql1_cols), len(sql2_cols)))
    for i in range(len(sql1_cols)):
        for j in range(len(sql2_cols)):
            if (
                sql1_cols[i].split(".")[-1].lower()
                == sql2_cols[j].split(".")[-1].lower()
            ):
                sim_matrix[i, j] = 1e3
                continue
            for agg_op in ["COUNT", "SUM", "AVG", "MAX", "MIN"]:
                if agg_op in sql1_cols[i] and agg_op in sql2_cols[j]:
                    sim_matrix[i, j] += 1
                    break
            vals1 = df_sql1.iloc[:, i].values
            vals2 = df_sql2.iloc[:, j].values
            score = len(set(vals1) & set(vals2))
            sim_matrix[i, j] += score
    # find the maximum score until the score is 0
    aligned_columns = []
    while True:
        (i, j) = np.unravel_index(np.argmax(sim_matrix), sim_matrix.shape)
        max_score = sim_matrix[i, j]
        sim_matrix[i, :] = 0
        sim_matrix[:, j] = 0
        if max_score == 0:
            break
        aligned_columns.append((sql_node1.exec_columns[i], sql_node2.exec_columns[j]))
    if len(aligned_columns) == max(
        len(sql_node1.exec_columns), len(sql_node2.exec_columns)
    ):
        return [], []
    sql1_aligned_cols = [col1 for col1, col2 in aligned_columns]
    sql2_aligned_cols = [col2 for col1, col2 in aligned_columns]
    sql1_unaligned_cols = [
        col for col in sql_node1.exec_columns if col not in sql1_aligned_cols
    ]
    sql2_unaligned_cols = [
        col for col in sql_node2.exec_columns if col not in sql2_aligned_cols
    ]
    return sql1_unaligned_cols, sql2_unaligned_cols


def gen_col_comparison_note(
    sql_node1: SQLNode,
    sql_node2: SQLNode,
    sql1_unaligned_cols: List[str],
    sql2_unaligned_cols: List[str],
) -> str:
    try:
        sql1_return_columns_desc_dict = json.loads(
            sql_node1.notes["returned_columns_note"]
        )
        sql2_return_columns_desc_dict = json.loads(
            sql_node2.notes["returned_columns_note"]
        )
        sql1_unaligned_cols_desc = "\n".join(
            [
                f"{col}: {desc}"
                for col, desc in sql1_return_columns_desc_dict.items()
                if col in sql1_unaligned_cols
            ]
        )
        sql2_unaligned_cols_desc = "\n".join(
            [
                f"{col}: {desc}"
                for col, desc in sql2_return_columns_desc_dict.items()
                if col in sql2_unaligned_cols
            ]
        )
    except:
        sql1_unaligned_cols_desc = "\n".join(sql1_unaligned_cols)
        sql2_unaligned_cols_desc = "\n".join(sql2_unaligned_cols)
    prompt = "Differences in returned columns:\n"
    if len(sql1_unaligned_cols) > 0:
        prompt += f"SQL1 returns the following columns that are not returned by SQL2:\n{sql1_unaligned_cols_desc}\n"
    if len(sql2_unaligned_cols) > 0:
        prompt += f"SQL2 returns the following columns that are not returned by SQL1:\n{sql2_unaligned_cols_desc}\n"
    prompt += "\nHint 1: Prefer ID over names if NL query does not explicitly ask for the name."
    prompt += "\nHint 2: Prefer the SQL that uses separate columns over concatenating them into one column."
    prompt += "\nHint 3: Reject UNIONs of semantically different columns—they must be distinct and meaningful."
    return prompt


def gen_group_by_note(
    sql_node1: SQLNode,
    sql_node2: SQLNode,
) -> str:
    # case 1: both not use GROUP BY clause
    if (
        "GROUP BY" not in sql_node1.org_sql.upper()
        or "GROUP BY" not in sql_node2.org_sql.upper()
    ):
        return None
    # # case 2: one use GROUP BY clause, the other not
    # if (
    #     "GROUP BY" in sql_node1.org_sql.upper()
    #     and "GROUP BY" not in sql_node2.org_sql.upper()
    # ):
    #     return f"SQL1 uses GROUP BY clause, but SQL2 does not. Please refer to the NL query to determine whether the GROUP BY clause is necessary."
    # if (
    #     "GROUP BY" in sql_node2.org_sql.upper()
    #     and "GROUP BY" not in sql_node1.org_sql.upper()
    # ):
    #     return f"SQL2 uses GROUP BY clause, but SQL1 does not. Please refer to the NL query to determine whether the GROUP BY clause is necessary."
    # case 3: both use GROUP BY clause
    sql1_group_by_cols, sql1_group_by_clause = extract_group_by(sql_node1.org_sql)
    sql2_group_by_cols, sql2_group_by_clause = extract_group_by(sql_node2.org_sql)
    if frozenset(sql1_group_by_cols) == frozenset(sql2_group_by_cols):
        return None
    return (
        f"SQL1 group by {sql1_group_by_clause},\nSQL2 group by {sql2_group_by_clause}."
    )


def gen_order_by_note(
    sql_node1: SQLNode,
    sql_node2: SQLNode,
) -> str:
    # classify the ranking in SQL into 5 cases: order by asc, order by desc, max, min, None
    sql1_ranking_type = classify_ranking(sql_node1.org_sql)
    sql2_ranking_type = classify_ranking(sql_node2.org_sql)
    # case 1: both not use ORDER BY clause, and no max/min aggregation
    if sql1_ranking_type is None or sql2_ranking_type is None:
        return None
    # different ranking order
    if (
        sql1_ranking_type == "order by asc"
        and sql2_ranking_type in ["order by desc", "max"]
    ) or (
        sql1_ranking_type == "order by desc"
        and sql2_ranking_type in ["order by asc", "min"]
    ):
        return f"Two SQLs rank the records in different orders. Please refer to the NL query to determine which order is correct."
    # same ranking order
    if (sql1_ranking_type == "order by asc" and sql2_ranking_type == "min") or (
        sql1_ranking_type == "order by desc" and sql2_ranking_type == "max"
    ):
        return order_vs_maxmin("SQL1", "SQL2")
    if (sql1_ranking_type == "min" and sql2_ranking_type == "order by asc") or (
        sql1_ranking_type == "max" and sql2_ranking_type == "order by desc"
    ):
        return order_vs_maxmin("SQL2", "SQL1")
    if sql1_ranking_type == sql2_ranking_type and sql1_ranking_type in [
        "order by desc",
        "max",
        "min",
    ]:
        pattern = r"CAST\s*\(\s*([\w.]+)\s*AS\s*REAL\s*\)"
        if_sql1_cast_to_real = re.search(pattern, sql_node1.org_sql, re.IGNORECASE)
        if_sql2_cast_to_real = re.search(pattern, sql_node2.org_sql, re.IGNORECASE)
        if if_sql1_cast_to_real and not if_sql2_cast_to_real:
            return f"SQL1 explicitly casts the order by object to the `REAL` type, while SQL2 does not. Please refer to the NL query to determine whether the order by object needs to be converted to the `REAL` type."
        if not if_sql1_cast_to_real and if_sql2_cast_to_real:
            return f"SQL2 explicitly casts the order by object to the `REAL` type, while SQL1 does not. Please refer to the NL query to determine whether the order by object needs to be converted to the `REAL` type."
        return None
    return "Both SQLs order the records in ascending order. If the ORDER BY object has null values, prefer the SQL that uses `NOT NULL` constraint on the ORDER BY object."


def gen_filtering_logic_note(
    sql_node1: SQLNode,
    sql_node2: SQLNode,
) -> str:
    # sql1_filtering_conditions_note = sql_node1.notes["filtering_conditions_note"]
    # sql2_filtering_conditions_note = sql_node2.notes["filtering_conditions_note"]
    # if sql1_filtering_conditions_note == sql2_filtering_conditions_note:
    #     return None
    # return f"SQL1 filtering conditions: {sql1_filtering_conditions_note}\nSQL2 filtering conditions: {sql2_filtering_conditions_note}"
    return "Prefer the SQL that returns all the records that satisfy the NL query."
