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


def compare_schema(
    base_info: str,
    sql_node1: SQLNode,
    sql_node2: SQLNode,
) -> str:
    """
    base_info =
    NL Query: {question}
    Evidence: {evidence}
    Schema: {schema_note}

    SQL1: {sql_node1.org_sql}
    SQL2: {sql_node2.org_sql}
    """

    prompt = f"""You are an expert at mapping natural language queries to database schemas. Your task is to determine which of two SQL queries uses **more appropriate tables and columns** to answer the given natural language question.

Important:  
- SQL1 and SQL2 use **different tables and/or columns**, and both SQLs are executable.  
- Table names and column names are not case-sensitive.
- Your evaluation **must focus exclusively** on whether the **tables and columns** referenced in each SQL logically correspond to the entities, attributes, and conditions expressed in the NL Query (and Evidence, if provided).  
- The relevance and completeness of the query logic (e.g., filters, joins, aggregations) with respect to the question.
- Prefer ID over names if NL query does not explicitly ask for the name.

Use the Schema to understand what each table and column represents.
The returned fields must strictly comply with the requirements of the NL query. 

{base_info}
Notes on the SQL1 Execution Results: {sql_node1.notes["exec_note"]}
Notes on the SQL2 Execution Results: {sql_node2.notes["exec_note"]}
"""
    if (
        sql_node1.notes["join_note"] is not None
        and len(sql_node1.notes["join_note"]) > 0
    ):
        prompt += f"Warnings on SQL1 Join Conditions: {sql_node1.notes['join_note']}\n"
    if (
        sql_node2.notes["join_note"] is not None
        and len(sql_node2.notes["join_note"]) > 0
    ):
        prompt += f"Warnings on SQL2 Join Conditions: {sql_node2.notes['join_note']}\n"

    prompt += """Output strictly in the following JSON format with no extra text:
{{
    "reason": "Explain specifically which tables/columns in the chosen SQL better satisfies the NL Query and why, referencing the Schema or Evidence.",
    "better_sql": "SQL1/SQL2/Both/Neither"
}}"""
    return prompt


def compare_top_n(
    base_info: str,
    sql_node1: SQLNode,
    sql_node2: SQLNode,
) -> str:
    prompt = f"""You are an expert in SQL and database reasoning. Your task is to determine which of two candidate SQL queries better answers the given natural language question, by explicitly comparing their differences and evaluating which aligns more closely with the evidence and schema. The two SQL queries produce different execution results. Therefore, at most one of them can be correct.

**Core Evaluation Principles**  
1. Correctness
1.1 The query must return exactly the set of rows that satisfy the natural language question and any provided evidence.
1.2 Generally, ORDER BY is used to sort results. To get the top N rows, always pair it with LIMIT (or TOP / FETCH FIRST as appropriate). 
1. Return All Tied Results Unless Specified Otherwise
1.1 If the natural language question asks for “the highest,” “the most,” or similar without specifying “one” or “a single,” all rows tied for the extremal value must be returned.
1.2 For top-N (e.g., “top 3”), use RANK() or DENSE_RANK() if ties should be included, and ROW_NUMBER() only if exactly N distinct rows are required (rare unless explicitly stated).
1.3 Correct approaches include:
- WHERE col = (SELECT MAX(col) FROM ...) — returns all ties,
- Window functions like RANK() combined with filtering (e.g., WHERE rnk = 1).consistent meaning across all rows.
2. Avoid Fragile Assumptions About NULLs in Ordering
2.1 While NULL values can affect sort order, do not penalize a query solely for not enforcing NOT NULL on ordering columns unless:
- The schema allows NULL in that column, and
- The presence of NULL would meaningfully distort the result (e.g., NULL treated as lowest/highest).
2.2 Focus instead on whether the ranking logic correctly reflects the intent given the data semantics.
3. Filter Before Ranking
- Ranking must be applied only to rows that satisfy the natural language condition.
4. Output Columns Must Match the Request Exactly
- If the question asks for an identifier (e.g., “which districts?”), returning the ID is acceptable.
- If it asks for a name or description, return the name column—not the ID.
- Prefer separate columns over concatenated strings (e.g., return first_name, last_name, not first_name || ' ' || last_name).
- Reject UNION of semantically unrelated columns (e.g., unioning product names with customer IDs)—each column must have 

Your Analysis Should Focus On:
1. Identify the key structural or semantic differences between SQL1 and SQL2 (e.g., different JOIN types, WHERE conditions, GROUP BY clauses, use of aggregation functions, subqueries, table aliases, or column selections).
2. For each difference, assess which version better matches:
   - The intent of the natural language question,
   - The constraints or facts in the evidence,
   - The relationships and semantics defined in the schema.
3. Conclude which query fully satisfies the natural language intent—including completeness (all ties), correctness (right entities), and fidelity (right columns)
Note: refer to Core Evaluation Principles for your analysis, and cite the principles in your analysis.

{base_info}
Notes on the SQL1 Execution Results: {sql_node1.notes["exec_note"]}
Notes on the SQL2 Execution Results: {sql_node2.notes["exec_note"]}

Output strictly in the following JSON format with no extra text:
{{
    "reason": "Briefly explain your reasoning process. Ensure your explanation is logically consistent.",
    "better_sql": "SQL1/SQL2/Unsure"
}}
"""
    return prompt


def compare_aggregation(
    base_info: str,
    sql_node1: SQLNode,
    sql_node2: SQLNode,
) -> str:
    prompt = f"""You are an expert in SQL and database reasoning. Your task is to determine which of two candidate SQL queries better answers the given natural language question, by explicitly comparing their differences and evaluating which aligns more closely with the evidence and schema. The two SQL queries produce different execution results. Therefore, at most one of them can be correct.

**Core Evaluation Principles**  
1. Filter Before Aggregate
1.1 Aggregation must be applied only to rows that satisfy the natural language condition.
1.2 Filtering should occur in WHERE (for row-level conditions) or via proper subqueries/CTEs.
1.3 Using HAVING without a corresponding GROUP BY—or using it to filter non-aggregated dimensions—is often incorrect unless the intent is to filter after grouping.
2. Grouping Must Match Query Granularity
2.1 If the question asks for “per X” (e.g., “average salary per department”), the query must include GROUP BY X (ID of X is preferred over name of X since the ID is more unique).
2.2 Missing or incorrect GROUP BY leads to either:
- Invalid SQL (in strict SQL modes), or
- Nonsensical results (e.g., arbitrary values for non-aggregated columns).
3. Avoid Non-Functional Dependencies in SELECT
3.1 In standard SQL, every column in SELECT must either:
- Be in the GROUP BY clause, or
- Be wrapped in an aggregate function (e.g., MAX, MIN), unless it is functionally dependent on the grouped key (e.g., a primary key uniquely determining other attributes).
3.2 Queries that select non-grouped, non-aggregated columns risk returning arbitrary or misleading values, even if they run without error.
4. Prevent Inflated Aggregates from Joins
4.1 If a JOIN creates a one-to-many relationship (e.g., joining orders to order_items), naive aggregation (e.g., SUM(price)) may double-count or overcount due to row duplication.
4.2 The better query should:
- Use DISTINCT inside aggregates when appropriate (e.g., COUNT(DISTINCT user_id)),
- Pre-aggregate in subqueries before joining, or
- Avoid unnecessary joins that distort cardinality—if the NL intent refers to unique entities.
5. Respect Explicit Formulas or Ratios from Evidence
5.1 If the evidence specifies a calculation (e.g., “ratio = number of defaults / total loans”), the SQL must replicate that exact formula.
5.2 Prefer numerator/denominator computed over the same filtered set (unless specified otherwise).
5.3 Be cautious of integer division: use CAST(... AS REAL) or equivalent when fractional results are expected.
6. Handle Edge Cases Gracefully
6.1 Division by zero (e.g., in ratios) should be avoided or handled explicitly if possible.
6.2 Empty result sets or NULL aggregates should align with expected behavior.
6.3 Unless explicitly specified to round to a certain number of decimal places, do not arbitrarily round numerical values.

Your Analysis Should Focus On:
1. Identify the key structural or semantic differences between SQL1 and SQL2 (e.g., different JOIN types, WHERE conditions, GROUP BY clauses, use of aggregation functions, subqueries, table aliases, or column selections).
2. For each difference, assess which version better matches:
   - The intent of the natural language question,
   - The constraints or facts in the evidence,
   - The relationships and semantics defined in the schema.
3. Consider whether a difference leads to inclusion/exclusion of relevant rows, incorrect aggregation, misinterpretation of entities, or violation of query intent.
Note: refer to Core Evaluation Principles for your analysis, and cite the principles in your analysis.

{base_info}
Notes on the SQL1 Execution Results: {sql_node1.notes["exec_note"]}
Notes on the SQL2 Execution Results: {sql_node2.notes["exec_note"]}
"""
    if (
        sql_node1.notes["join_note"] is not None
        and len(sql_node1.notes["join_note"]) > 0
    ):
        prompt += f"Warnings on SQL1 Join Conditions: {sql_node1.notes['join_note']}\n"
    if (
        sql_node2.notes["join_note"] is not None
        and len(sql_node2.notes["join_note"]) > 0
    ):
        prompt += f"Warnings on SQL2 Join Conditions: {sql_node2.notes['join_note']}\n"

    prompt += """
Output strictly in the following JSON format with no extra text:
{{
    "reason": "Briefly explain your reasoning process. Ensure your explanation is logically consistent. Refer to Core Evaluation Principles for your analysis, and cite the principles in your analysis.",
    "better_sql": "SQL1/SQL2/Unsure"
}}
"""
    return prompt


def compare_other(
    base_info: str,
    sql_node1: SQLNode,
    sql_node2: SQLNode,
) -> str:
    prompt = f"""You are an expert in SQL and database reasoning. Your task is to determine which of two candidate SQL queries better answers the given natural language question, by explicitly comparing their differences and evaluating which aligns more closely with the evidence, schema, and query intent. The two SQL queries produce different execution results. Therefore, at most one of them can be correct.

Core Evaluation Principles:
1. Correctness & Completeness:
1.1 The query must return exactly the set of rows that satisfy the natural language question and any provided evidence.
1.2 If multiple rows tie for a ranking condition (e.g., “highest,” “most”), all tied rows must be returned.
1.3 Prefer queries that use MAX()/MIN() in a filtering condition (e.g., WHERE col = (SELECT MAX(col) ...)) over ORDER BY ... LIMIT 1 when ties are possible, because the latter arbitrarily drops tied results.
1.4 Use RANK() (not ROW_NUMBER()) when explicit ranking with ties is needed.
2. Schema & Semantic Fidelity:
2.1 Column and table names must match the schema in spelling and semantics (case-insensitive matching is assumed).
2.2 COUNT(*) counts rows; COUNT(col) counts non-NULL values in col. Choose based on whether NULLs should be excluded.
2.3 Refer to the schema to understand the meaning of the columns and tables.
3. Output Alignment:
3.1 Return the columns explicitly requested in the natural language question.
3.2 Prefer separate columns over concatenated strings.
3.3 If the question asks for an identifier (e.g., “which districts”), returning the ID is acceptable unless a name is explicitly requested.
3.4 Unless explicitly specified to round to a certain number of decimal places, do not arbitrarily round numerical values.
4. Query Robustness:
4.1 Avoid UNION of semantically unrelated columns.
4.2 Ensure filters (e.g., year extraction, category codes) exactly match the evidence and question.
4.3 When computing numbers (e.g., average, count), ensure the logic matches the intent. When dividing numbers, prefer to cast the numerator and denominator to the `REAL` type.

Your Analysis Should Focus On:
1. Identify key differences between SQL1 and SQL2: Filtering conditions (WHERE), joins, aggregation (GROUP BY, COUNT, SUM), subqueries, ranking logic, selected columns, etc.
2. For each difference, assess:
- Does it affect correctness (wrong rows included/excluded)?
- Does it violate output requirements (missing/wrong columns)?
- Does it fail to handle ties or edge cases (e.g., NULLs, multiple maxima)?
- Is it inconsistent with the schema or evidence?
Note: refer to Core Evaluation Principles for your analysis, and cite the principles in your analysis.
3. Conclude which query fully satisfies the natural language intent, evidence, and schema.

{base_info}
Notes on the SQL1 Execution Results: {sql_node1.notes["exec_note"]}
Notes on the SQL2 Execution Results: {sql_node2.notes["exec_note"]}
"""
    if (
        sql_node1.notes["join_note"] is not None
        and len(sql_node1.notes["join_note"]) > 0
    ):
        prompt += f"Warnings on SQL1 Join Conditions: {sql_node1.notes['join_note']}\n"
    if (
        sql_node2.notes["join_note"] is not None
        and len(sql_node2.notes["join_note"]) > 0
    ):
        prompt += f"Warnings on SQL2 Join Conditions: {sql_node2.notes['join_note']}\n"

    prompt += """
Output strictly in the following JSON format with no extra text:
{{
    "reason": "Briefly explain your reasoning process. Ensure your explanation is logically consistent.",
    "better_sql": "SQL1/SQL2/Unsure"
}}
"""
    return prompt
