from parser import SQLCollection, SQLNode
from utils import execute_sql, execute_sql_wrapper
from typing import Dict, List, Any, Tuple
from database import Database
from sqlglot import parse_one, exp
from analyzer import (
    understand_sql,
    collect_schema_info,
    llm_check,
    get_exec_examples,
    form_prompt,
)
import json
import re
from syntax import generate_join_note
from execution import generate_exec_note
from collections import defaultdict
import pandas as pd


def create_view(
    views: Dict[str, str], db: Database, table_columns: Dict[str, List[str]]
) -> Tuple[str, str]:
    view_notes = []
    for view_name, view_sql in views.items():
        sql = f"DROP VIEW IF EXISTS {view_name}"
        exec_res = execute_sql(sql, db.db_path)
        sql = f"CREATE VIEW {view_name} AS {view_sql}"
        print(f"[view sql]: {sql}")
        exec_res = execute_sql_wrapper(sql, db.db_path, 10)
        # test retrieving data from view
        sql = f"SELECT * FROM {view_name} limit 3"
        exec_res = execute_sql_wrapper(sql, db.db_path, 10)
        if exec_res == "Time Out" or exec_res == "Unexecutable":
            return "Failed", ""
        # update the db with the view
        db.add_view(view_name)
        print(f"[view]: {view_name}")
        # TODO: generate the note for the view
        exec_note = get_exec_examples(exec_res)
        schema_note = collect_schema_info(table_columns, db)
        prompt = form_prompt(
            None,
            None,
            view_sql,
            exec_note,
            schema_note,
            None,
            False,
            "\n".join(view_notes),
        )
        note = llm_check([prompt])[0]
        sql = f"CREATE VIEW {view_name} AS {view_sql}"
        view_notes.append(f"View: {view_name}\nSQL: {sql}\nNote: {note}\n")
    return "Done", "\n".join(view_notes)


def generate_returned_columns_prompt(
    question: str, evidence: str, schema_note: str, sql_node: SQLNode
) -> str:
    # prompt for asking the LLM to explain each returned column
    prompt = f"""You are given a natural language query (NL Query), optional supporting evidence that clarifies the intent of the query, the database schema, the column names returned by a SQL query and the SQL query itself. Your task is to briefly describe each returned column with respect to the NL Query, evidence and schema.
    
Input Format:
NL Query: {question}
Evidence: {evidence}
Schema: {schema_note}
SQL: {sql_node.org_sql}
Returned Columns: {sql_node.exec_columns}

Output Format:
{{
    "column1": "Brief description of column1",
    "column2": "Brief description of column2",
    ...
}}
"""
    return prompt


def generate_evidence_alignment_prompt(
    question: str, evidence: str, schema_note: str, sql_node: SQLNode
) -> str:
    # prompt for asking the LLM to identify the parts in the SQL that is relevant to the evidence, and determine whether the operation in the SQL is consistent with the evidence.
    prompt = f"""You are given a natural language query (NL Query), optional supporting evidence that clarifies the intent of the query, the database schema, the column names returned by a SQL query and the SQL query itself. Your task is to identify the parts in the SQL that is relevant to the evidence, and determine whether the operation in the SQL is consistent with the evidence.

You should follow the two steps:
1. Identify the parts in the SQL that is relevant to the evidence.
2. Determine whether the operation in the SQL is consistent with the evidence.
    
Input Format:
NL Query: {question}
Evidence: {evidence}
Schema: {schema_note}
SQL: {sql_node.org_sql}

Output Format:
{{
    "analysis": "Briefly explain your reasoning process.",
    "consistent": "Yes/No/Unsure",
}}
"""
    return prompt


def generate_filtering_conditions_prompt(
    question: str, evidence: str, schema_note: str, sql_node: SQLNode
) -> str:
    # prompt for asking the LLM to identify the filtering conditions, filtering condition separated by "AND", and explain each filtering condition by referring to the NL Query, evidence and schema.
    prompt = f"""You are given a natural language query (NL Query), optional supporting evidence that clarifies the intent of the query, the database schema, the column names returned by a SQL query and the SQL query itself. Your task is to identify the filtering conditions, filtering condition separated by "AND", and explain each filtering condition by referring to the NL Query, evidence and schema.
    
Input Format:
NL Query: {question}
Evidence: {evidence}
Schema: {schema_note}
SQL: {sql_node.org_sql}

Output Format:
{{
    "Condition 1": "Briefly explain the filtering condition 1",
    "Condition 2": "Briefly explain the filtering condition 2",
    ...
}}
"""
    return prompt


def pointwise_selection(
    sql_collection: SQLCollection, question: str, evidence: str
) -> List[SQLNode]:
    sql_nodes = sql_collection.sql_nodes
    table_columns = sql_collection.table_columns
    db = sql_collection.db
    schema_note = collect_schema_info(table_columns, db)
    returned_columns_prompts = []
    evidence_alignment_prompts = []
    filtering_conditions_prompts = []
    for sql_node in sql_nodes:
        # update the tables and columns for each node
        for node in sql_node.subsql_nodes:
            node.update_tables_and_columns()
            node.update_if_independent(db)
        # syntax check: exec, join, group by
        # TODO: modify the execution notes
        # generate notes for execution results
        exec_note, exec_warning_cnt = generate_exec_note(sql_node, db)
        # generate notes for join conditions
        join_note, join_warning_cnt = generate_join_note(sql_node, db)
        # get prompt for returned columns
        returned_columns_prompts.append(
            generate_returned_columns_prompt(question, evidence, schema_note, sql_node)
        )
        evidence_alignment_prompts.append(
            generate_evidence_alignment_prompt(
                question, evidence, schema_note, sql_node
            )
        )
        filtering_conditions_prompts.append(
            generate_filtering_conditions_prompt(
                question, evidence, schema_note, sql_node
            )
        )
        sql_node.warning_cnt = (
            exec_warning_cnt + join_warning_cnt
        )  # + intent_warning_cnt
        sql_node.notes = {
            "exec_note": exec_note,
            "join_note": join_note,
            # "intent_note": intent_note,
        }
    # collectively generate the notes for returned columns
    returned_columns_notes = llm_check(returned_columns_prompts)
    if evidence is not None and len(evidence.strip()) > 0:
        evidence_alignment_notes = llm_check(evidence_alignment_prompts)
    else:
        evidence_alignment_notes = [None] * len(sql_nodes)
    filtering_conditions_notes = llm_check(filtering_conditions_prompts)
    for (
        sql_node,
        returned_columns_note,
        evidence_alignment_note,
        filtering_conditions_note,
    ) in zip(
        sql_nodes,
        returned_columns_notes,
        evidence_alignment_notes,
        filtering_conditions_notes,
    ):
        sql_node.notes["returned_columns_note"] = returned_columns_note
        if evidence_alignment_note is not None:
            sql_node.notes["evidence_alignment_note"] = evidence_alignment_note
        sql_node.notes["filtering_conditions_note"] = filtering_conditions_note
    return sql_nodes


def get_col_comparison_prompt(
    question: str,
    evidence: str,
    schema_note: str,
    sql_node1: SQLNode,
    sql_node2: SQLNode,
) -> str:
    prompt = f"""You are given a natural language (NL) query, optional supporting evidence that clarifies the intent of the query, the database schema, and the column names returned by two candidate SQL queries (SQL1 and SQL2).

Your task is to determine which SQL query returns a more appropriate set of columns with respect to the NL query and evidence. Your goal is to identify ABSOLUTELY UNREASONABLE cases, not to choose the slightly better one when both are reasonable. If both SQLs return reasonable columns, return "Both". If neither is reasonable, return "Neither"; otherwise, return the more appropriate SQL query.

Guidelines:
- Prefer keeping the separate columns to concatenating them into one column unless explicitly required by the NL query.
- Reject UNIONs of semantically different columns—they must be distinct and meaningful.
- If the NL query does not explicitly ask for the name, prefer ID over names.
- It is acceptable to return more columns than the NL query asks for.

Input Format:
NL Query: {question}
Evidence (optional): {evidence}
Schema: {schema_note}

SQL1 returns columns: {sql_node1.exec_columns}
SQL1: {sql_node1.org_sql}
SQL2 returns columns: {sql_node2.exec_columns}
SQL2: {sql_node2.org_sql}

Ouput Format:
{{
    "reason": "Briefly explain your reasoning process on whether the returned columns of SQL1 and SQL2 are appropriate or not, respectively.",
    "better_sql": "SQL1/SQL2/Both/Neither",
}}
"""
    return prompt


def get_evidence_alignment_prompt(
    question: str,
    evidence: str,
    schema_note: str,
    sql_node1: SQLNode,
    sql_node2: SQLNode,
) -> str:
    prompt = f"""You are a strict consistency checker. Your task is to determine whether each of the two SQL queries is consistent with the given evidence, based only on what is explicitly stated in the evidence. Do not use external knowledge, assumptions about the NL query, or infer unstated constraints.

Follow these rules precisely:
1. Evidence is the sole source of truth for additional constraints on the NL query. Ignore anything not explicitly mentioned in the evidence.
2. A SQL query is inconsistent with the evidence only if:
   - It contradicts a fact, condition, calculation method, or logical rule explicitly stated in the evidence; OR
   - It uses a different aggregation, filter, join logic, or formula that directly conflicts with an explicit instruction in the evidence.
3. If the evidence mentions something that the SQL does not need to use, that is not a conflict.

Your task is to: Determine which SQL query is more consistent with the evidence. If both are consistent, return "both"; if neither is consistent, return "neither"; otherwise, return the more consistent SQL query.

Input Format:
NL Query: {question}
Evidence: {evidence}
Schema: {schema_note}
SQL1: {sql_node1.org_sql}
SQL2: {sql_node2.org_sql}

Ouput Format:
{{
    "reason": "Check the consistency of the SQL queries with the evidence, respectively.",
    "better_sql": "SQL1/SQL2/Both/Neither",
}}
"""
    return prompt


def get_filtering_conditions_prompt(
    question: str,
    evidence: str,
    schema_note: str,
    sql_node1: SQLNode,
    sql_node2: SQLNode,
) -> str:
    prompt = f"""Your task is to evaluate the correctness and appropriateness of filtering conditions in two SQL queries (SQL1 and SQL2) based on the provided NL Query, evidence and schema. Focus specifically on the filtering conditions in both SQL queries and determine their correctness by following these steps:

1. If the filtering conditions in both SQL queries are completely identical (same columns and conditions), return both.
2. If they differ, analyze the differences by considering the following cases and return the more appropriate SQL query:
Case 1: Both queries filter on the same column, but use different conditions (e.g., price > 100 vs. price > 200). Determine which condition aligns better with the NL query and evidence.
Case 2: Both queries use the same filter value or logic, but apply it to different columns (e.g., category = 'Electronics' vs. department = 'Electronics'). Assess which column is more appropriate based on the schema, NL intent and evidence.
Case 3: A filtering condition appears in only one of the queries. Evaluate whether this condition is necessary (i.e., implied by the NL query) or extraneous (adds incorrect or unsupported constraints).

Input Format:
NL Query: {question}
Evidence: {evidence}
Schema: {schema_note}
SQL1: {sql_node1.org_sql}
SQL2: {sql_node2.org_sql}

Ouput Format:
{{
    "reason": "Briefly explain your reasoning process.",
    "better_sql": "SQL1/SQL2/Both/Neither",
}}
"""
    return prompt


def get_overall_comparison_prompt(
    question: str,
    evidence: str,
    schema_note: str,
    sql_node1: SQLNode,
    sql_node2: SQLNode,
) -> str:
    # compare the two SQLs with different execution results
    prompt = f"""You are given a natural language query (NL Query) and two SQL statements that produce different results when executed. Analyze and determine which SQL is more aligned with the intent of the NL Query. Identify key differences in operations between the two SQL statements (e.g., filtering conditions, sorting, aggregation) and focus on the aspects that lead to the different results. Analyze which SQL appears more reliable based on these differences.

Additional guidelines:
1. If the NL Query involves finding the "most XX" (e.g., maximum, minimum), prioritize SQL that uses MAX/MIN as it is more robust than using ORDER BY with LIMIT, especially in tied scenarios. Next, consider SQL with ORDER BY + LIMIT, ensuring the presence of LIMIT to guarantee a unique result.
2. If the NL Query asks for the rank of the result, prefer RANK() over ROW_NUMBER() as it is more appropriate for ranking.
3. Prefer ID over names if NL query does not explicitly ask for the name.
4. Prefer the SQL that uses less tables if the information is sufficient.
5. Prefer the SQL that uses separate columns over concatenating them into one column unless explicitly required by the NL query.
6. Reject UNIONs of semantically different columns—they must be distinct and meaningful.

**NL query:**
{question}
**Schema information:**
{schema_note}
**Evidence:**
{evidence}

**SQL1:**
[SQL]: 
{sql_node1.org_sql}
[Execution Result]:
{sql_node1.notes["exec_note"]}
[Join Note]:
{sql_node1.notes["join_note"]}

**SQL2:**
[SQL]: 
{sql_node2.org_sql}
[Execution Result]:
{sql_node2.notes["exec_note"]}
[Join Note]:
{sql_node2.notes["join_note"]}

Please select the better SQL query for the NL query. Your response must be in the following JSON format:
{{
    "reason": "Briefly explain your reasoning process.",
    "better_sql": "SQL1/SQL2/Unsure",
}}
"""
    return prompt


def multi_aspect_sql_comparison(
    question: str,
    evidence: str,
    schema_note: str,
    sql_node1: SQLNode,
    sql_node2: SQLNode,
) -> Tuple[str, str]:
    prompt_dict = {}
    # Aspect 1: returned columns (Final SELECT Clause)
    if sql_node1.exec_columns != sql_node2.exec_columns:
        col_comparison_prompt = get_col_comparison_prompt(
            question, evidence, schema_note, sql_node1, sql_node2
        )
        prompt_dict["col_comparison"] = col_comparison_prompt
    # Aspect 2: Evidence Alignment (WHERE Clause or Computation)
    if evidence is not None and len(evidence.strip()) > 0:
        evidence_alignment_prompt = get_evidence_alignment_prompt(
            question, evidence, schema_note, sql_node1, sql_node2
        )
        prompt_dict["evidence_alignment"] = evidence_alignment_prompt
    # Aspect 3: Filtering Conditions (WHERE Clause)
    filtering_conditions_prompt = get_filtering_conditions_prompt(
        question, evidence, schema_note, sql_node1, sql_node2
    )
    prompt_dict["filtering_conditions"] = filtering_conditions_prompt
    # Aspect 4: Overall Comparison
    overall_comparison_prompt = get_overall_comparison_prompt(
        question, evidence, schema_note, sql_node1, sql_node2
    )
    prompt_dict["overall_comparison"] = overall_comparison_prompt
    aspects, prompts = zip(*prompt_dict.items())
    results = llm_check(prompts)
    score = 0
    aspect_results = {}
    for aspect, result in zip(aspects, results):
        aspect_results[aspect] = result
        score += parse_result(result)
    if score > 0:
        better_sql = "SQL1"
    elif score < 0:
        better_sql = "SQL2"
    else:
        better_sql = "Unsure"
    return better_sql, aspect_results


def parse_result(result: str) -> int:
    start_idx = result.find("{")
    end_idx = result.rfind("}") + 1
    result = result[start_idx:end_idx]
    try:
        result = json.loads(result)
        if result["better_sql"] == "SQL1":
            return 1
        elif result["better_sql"] == "SQL2":
            return -1
        else:
            return 0
    except:
        return 0


def compare_sql(
    question: str,
    evidence: str,
    schema_note: str,
    selected_sql: SQLNode,
    sql_node: SQLNode,
) -> str:
    # compare the two SQLs with different execution results, and select the better one
    if len(selected_sql.exec_res) == 0 and len(sql_node.exec_res) == 0:
        return selected_sql
    if len(selected_sql.exec_res) == 0:
        return sql_node
    if len(sql_node.exec_res) == 0:
        return selected_sql
    prompt = get_overall_comparison_prompt(
        question, evidence, schema_note, selected_sql, sql_node
    )
    res = llm_check([prompt])[0]
    start_idx = res.find("{")
    end_idx = res.rfind("}") + 1
    res = res[start_idx:end_idx]
    try:
        res = json.loads(res)
        better_sql = res["better_sql"]
        print(res)
        print("-" * 100)
    except:
        better_sql = "Unsure"
    # selected_node = sql_node if better_sql == "SQL2" else selected_sql
    comparison_note = res
    return better_sql, comparison_note


def collective_selection(
    sql_collection: SQLCollection,
    question: str,
    evidence: str,
    sorted_sql_nodes: List[SQLNode],
) -> str:
    if len(sql_collection.sql_nodes) == 0:
        return "Error SQL"
    checked_sql_nodes = []
    # select the sql nodes with comparison
    selected_sql = sorted_sql_nodes[0]
    comparison_notes = []
    while True:
        cand_sql_node = None
        for i, sql_node in enumerate(sorted_sql_nodes):
            if (
                sql_node not in checked_sql_nodes
                and sql_node.org_sql not in selected_sql.covered_sqls
                and sql_node.org_sql != selected_sql.org_sql
            ):
                cand_sql_node = sql_node
                break
        if cand_sql_node is None:
            break
        # compare the selected sql with the cand sql node
        schema_note = collect_schema_info(
            sql_collection.table_columns, sql_collection.db
        )
        sql1 = selected_sql.org_sql
        sql2 = cand_sql_node.org_sql
        better_sql, comparison_note = multi_aspect_sql_comparison(
            question, evidence, schema_note, selected_sql, cand_sql_node
        )
        comparison_notes.append(
            {"SQL1": sql1, "SQL2": sql2, "comparison_note": comparison_note}
        )
        checked_sql_nodes.append(cand_sql_node)
        if better_sql == "SQL2":
            selected_sql = cand_sql_node
    return selected_sql.org_sql, comparison_notes


def collective_selection_old(
    sql_collection: SQLCollection,
    question: str,
    evidence: str,
    grouped_sql_nodes: List[List[SQLNode]],
) -> str:
    if len(sql_collection.sql_nodes) == 0:
        return "Error SQL"
    # sort the sql nodes by the warning cnt in ascending order
    sql_nodes = sorted(sql_collection.sql_nodes, key=lambda x: x.warning_cnt)
    # select the sql nodes with comparison
    selected_sql = sql_nodes[0]
    if selected_sql.warning_cnt >= 10:
        return "Error SQL"
    for sql_node in sql_nodes[1:]:
        selected_sql_exec_res = frozenset(selected_sql.exec_res)
        if sql_node.warning_cnt > selected_sql.warning_cnt + 1:
            break
        if selected_sql_exec_res == frozenset(sql_node.exec_res):
            continue
        # compare the SQL with different execution results
        schema_note = collect_schema_info(
            sql_collection.table_columns, sql_collection.db
        )
        better_sql, comparison_note = compare_sql(
            question, evidence, schema_note, selected_sql, sql_node
        )
    return selected_sql.org_sql


def align_exec_res(exec_res, cand_exec_res_list):
    if len(cand_exec_res_list) == 0:
        return None
    for cand_exec_res in cand_exec_res_list:
        if_align = if_align_exec_res_with_gt(exec_res, cand_exec_res)
        if if_align:
            return cand_exec_res
    return None


def if_align_exec_res_with_gt(exec_res, gt_exec_res) -> bool:
    if len(exec_res) != len(gt_exec_res):
        return False
    df_gt = pd.DataFrame(
        gt_exec_res, columns=[f"col_{i}" for i in range(len(gt_exec_res[0]))]
    )
    df_exec = pd.DataFrame(
        exec_res, columns=[f"col_{i}" for i in range(len(exec_res[0]))]
    )
    column_order = []
    for i in range(len(gt_exec_res[0])):
        # find the most similar column in df_exec to the i-th column of df_gt
        most_similar_column = None
        most_similar_score = 0
        for j in range(len(exec_res[0])):
            vals1 = df_gt.iloc[:, i].values
            vals2 = df_exec.iloc[:, j].values
            score = len(set(vals1) & set(vals2)) / len(set(vals1) | set(vals2))
            if score > most_similar_score and f"col_{j}" not in column_order:
                most_similar_score = score
                most_similar_column = j
        if most_similar_column is not None and most_similar_score == 1:
            column_order.append(f"col_{most_similar_column}")
        else:
            return False
    return True


def rank_sql_nodes(sql_nodes: List[SQLNode], sql_cnt: Dict[str, int]) -> List[SQLNode]:
    # sort the sqls by the number of columns in the execution results
    sql_nodes = sorted(sql_nodes, key=lambda x: len(x.exec_res[0]))
    # group the sql nodes by the execution results
    sql_covered_sqls = defaultdict(list)
    sql_coverage = defaultdict(int)
    for i, sql_node1 in enumerate(sql_nodes):
        sql_coverage[sql_node1.org_sql] += sql_cnt.get(sql_node1.org_sql, 1)
        for j, sql_node2 in enumerate(sql_nodes):
            if i >= j:
                continue
            exec_res1 = sql_node1.exec_res
            exec_res2 = sql_node2.exec_res
            if frozenset(exec_res1) == frozenset(exec_res2):
                sql_covered_sqls[sql_node1.org_sql].append(sql_node2.org_sql)
                sql_covered_sqls[sql_node2.org_sql].append(sql_node1.org_sql)
                sql_coverage[sql_node1.org_sql] += sql_cnt.get(sql_node2.org_sql, 1)
                sql_coverage[sql_node2.org_sql] += sql_cnt.get(sql_node1.org_sql, 1)
            elif len(exec_res1) == len(exec_res2):
                if len(exec_res1[0]) < len(exec_res2[0]):
                    # check if exec_res2 covers exec_res1
                    if_align = if_align_exec_res_with_gt(exec_res2, exec_res1)
                    if if_align:
                        sql_covered_sqls[sql_node2.org_sql].append(sql_node1.org_sql)
                        sql_coverage[sql_node2.org_sql] += sql_cnt.get(
                            sql_node1.org_sql, 1
                        )
                elif len(exec_res1[0]) > len(exec_res2[0]):
                    # check if exec_res1 covers exec_res2
                    if_align = if_align_exec_res_with_gt(exec_res1, exec_res2)
                    if if_align:
                        sql_covered_sqls[sql_node1.org_sql].append(sql_node2.org_sql)
                        sql_coverage[sql_node1.org_sql] += sql_cnt.get(
                            sql_node2.org_sql, 1
                        )
        # print(f"[DEBUG][sql_coverage]: {sql_coverage[sql_node1.org_sql]}")
    for sql_node in sql_nodes:
        sql_node.covered_sqls = sql_covered_sqls.get(sql_node.org_sql, [])
        sql_node.coverage = sql_coverage.get(sql_node.org_sql, 0)
    sorted_sql_nodes = sorted(
        sql_nodes,
        key=lambda x: (
            x.warning_cnt,
            -x.coverage,
            x.exec_time,
        ),
    )
    for i, sql_node in enumerate(sorted_sql_nodes):
        sql_node.rank = f"{i}-{sql_node.coverage}"
    return sorted_sql_nodes


def exhaustive_pairwise(
    sql_collection: SQLCollection,
    question: str,
    evidence: str,
    sql_nodes: List[SQLNode],
):
    comparison_notes = []
    votes = [0] * len(sql_nodes)
    for i, sql_node1 in enumerate(sql_nodes):
        for j, sql_node2 in enumerate(sql_nodes):
            if i >= j:
                continue
            if frozenset(sql_node1.exec_res) == frozenset(sql_node2.exec_res):
                votes[i] += 0.5
                votes[j] += 0.5
                continue
            sql1 = sql_node1.org_sql
            sql2 = sql_node2.org_sql
            schema_note = collect_schema_info(
                sql_collection.table_columns, sql_collection.db
            )
            better_sql, comparison_note = compare_sql(
                question, evidence, schema_note, sql_node1, sql_node2
            )
            comparison_notes.append(
                {"SQL1": sql1, "SQL2": sql2, "comparison_note": comparison_note}
            )
            if better_sql == "SQL1":
                votes[i] += 1
            elif better_sql == "SQL2":
                votes[j] += 1
            else:
                votes[i] += 0.5
                votes[j] += 0.5
    selected_sql = sql_nodes[votes.index(max(votes))]
    return selected_sql.org_sql, comparison_notes


def filter_by_returned_columns(
    sql_collection: SQLCollection,
    question: str,
    evidence: str,
) -> List[SQLNode]:
    candidate_columns = sql_collection.sql_columns
    column_sqls = defaultdict(list)
    for sql, columns in candidate_columns.items():
        column_sqls[frozenset(columns)].append(sql)
    if len(column_sqls) == 1:
        return sql_collection.sql_nodes
    candidate_columns_formatted = "\n".join(
        [f"C{i}: {list(columns)}" for i, columns in enumerate(column_sqls.keys())]
    )
    prompt = f"""You are an expert in natural language-to-SQL semantic validation.

Given a natural language (NL) query, evidence (if any), and multiple candidate column sets (each labeled C1, C2, etc., representing the SELECTed columns of a SQL query), perform the following:

1. **Infer the required output attributes**:  
   Identify only the attributes explicitly requested as output in the NL query (e.g., "list the names", "show the salary", "return department and count").
   Do not include attributes that are only used for filtering, joining, grouping, or sorting (e.g., conditions like "where score > 100" or "enrolled after 2020" are not output attributes unless the query also asks to "show the score" or "display the enrollment year").

2. **Evaluate each candidate**:  
   For each candidate set, determine if it contains at least one semantically relevant column for every required output attribute identified in Step 1.
   Semantic Relevance Over Lexical Match: Judge based on meaning, not exact string matching. For example, School, SchoolName, Institution, and Name can all correspond to the output attribute "school name". Count, Total, COUNT(*) can correspond to a "number" or "count" attribute.
   Extra columns are allowed.
   A candidate is invalid only if it lacks a column semantically relevant to a required output attribute.

3. **Summarize only the invalid candidates**:  
   In your output, list **only the candidates that are invalid**, along with the missing required fields for each.

4. **Output format**:  
   Return a JSON object with:
   - `"required_fields"`: list of required semantic attributes (in plain English)
   - `"invalid_candidates"`: a dictionary mapping candidate labels (e.g., "C1") to a list of missing required fields

---
Natural Language Query: {question}
Evidence: {evidence}
Candidate Column Sets:
{candidate_columns_formatted}
"""
    print(f"[DEBUG][prompt]:\n{prompt}")
    res = llm_check([prompt])[0]
    start_idx = res.find("{")
    end_idx = res.rfind("}") + 1
    res = res[start_idx:end_idx]
    try:
        res = json.loads(res)
        required_fields = res["required_fields"]
        invalid_candidates = res["invalid_candidates"]
        print("[DEBUG][res]:--------------------------------")
        print(res)
        print("-" * 100)
    except:
        required_fields = []
        invalid_candidates = {}
    filtered_sqls = []
    for i, columns in enumerate(column_sqls.keys()):
        if f"C{i}" not in invalid_candidates:
            filtered_sqls.extend(column_sqls[columns])
    # filtered sql nodes
    filtered_sql_nodes = []
    for sqlnode in sql_collection.sql_nodes:
        if sqlnode.org_sql in filtered_sqls:
            filtered_sql_nodes.append(sqlnode)
    sql_collection.sql_nodes = filtered_sql_nodes
    return filtered_sql_nodes
