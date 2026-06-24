from parser import SQLCollection, SQLNode
from utils import execute_sql, execute_sql_wrapper, parse_result, parse_json
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
from syntax import generate_join_note, compare_tb_cols
from execution import generate_exec_note
from collections import defaultdict
import pandas as pd
import numpy as np
from comparison_note import (
    same_returned_columns,
    gen_col_comparison_note,
    gen_group_by_note,
    gen_order_by_note,
    gen_filtering_logic_note,
    compare_schema,
    compare_top_n,
    compare_aggregation,
    compare_other,
)
from aspect import Aspect
from prompts import (
    intra_group_selection_prompt,
    get_comparison_prompt,
    get_simple_comparison_prompt,
    get_rule_evaluation_prompt,
    get_generative_verifier_prompt,
)
from binary_comparison import get_prompt
from rule_lib import RuleCollection


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
    prompt = f"""You are given the database schema, optional supporting evidence, the column names returned by a SQL query and the SQL query itself. Your task is to briefly describe each returned column with respect to the NL Query, evidence and schema.
    
Input Format:
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
    prompt = f"""You are given an evidence statement, a database schema, and a SQL query. Your task is to identify only the parts that are mutually relevant between the evidence and the SQL, and assess their consistency.

Guidelines:
1. The evidence provides supplementary context—it does not need to mention everything in the SQL, and the SQL does not need to include everything from the evidence.
2. Only when the SQL and the evidence refer to the same fact or condition, they must not contradict each other.
3. Focus your analysis only on parts of the SQL that correspond to claims or details explicitly stated in the evidence.
4. Extract the claims from the evidence and the relevant parts from the SQL, and then assess their consistency.
5. If the evidence specifies the filtering conditions, the SQL should have the same filtering conditions.
6. Numerical and computational precision matters: Pay close attention to how quantities are defined. Any mismatch in such definitions constitutes a logical inconsistency.
    
Input Format:
Evidence: {evidence}
Schema: {schema_note}
SQL: {sql_node.org_sql}

Output Format:
A list of aligned claim pairs in the following format:
[
{{
"evidence_claim": "claim extracted from evidence",
"sql_clause": "corresponding SQL clause extracted from SQL",
"consistency": "Yes/No/Unsure"
}},
...
]
"""
    # "analysis": "briefly summarize the inconsistency if any, otherwise, consistent with the evidence",
    return prompt


def generate_filtering_conditions_prompt(
    question: str, evidence: str, schema_note: str, sql_node: SQLNode
) -> str:
    # prompt for asking the LLM to identify the filtering conditions, filtering condition separated by "AND", and explain each filtering condition by referring to the NL Query, evidence and schema.
    prompt = f"""You are given the database schema, optional supporting evidence and the SQL query. Your task is to extract each atomic filtering condition from the WHERE clause of the SQL query (split by AND), and for each one, provide a clear, concise and brief description by referring to the evidence and schema. Your output should be a json dictionary with the extracted condition from the SQL as the key and the brief description as the value. For example, {{"num > 100": "The number of the item is greater than 100"}}.
    
Input Format:
Schema: {schema_note}
Evidence: {evidence}
SQL: {sql_node.org_sql}

Output Format:
{{
    "extracted_condition_1": "Brief description of the extracted condition 1 from the SQL",
    "extracted_condition_2": "Brief description of the extracted condition 2 from the SQL",
    ...
}}
"""
    return prompt


def syntax_level_selection(
    sql_collection: SQLCollection, question: str, evidence: str
) -> List[SQLNode]:
    sql_nodes = sql_collection.sql_nodes
    table_columns = sql_collection.table_columns
    db = sql_collection.db
    schema_note = collect_schema_info(table_columns, db)
    returned_columns_prompts = []
    evidence_alignment_prompts = []
    filtering_conditions_prompts = []
    filtred_sql_nodes = []
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
        # # get prompt for returned columns
        # returned_columns_prompts.append(
        #     generate_returned_columns_prompt(question, evidence, schema_note, sql_node)
        # )
        # evidence_alignment_prompts.append(
        #     generate_evidence_alignment_prompt(
        #         question, evidence, schema_note, sql_node
        #     )
        # )
        # filtering_conditions_prompts.append(
        #     generate_filtering_conditions_prompt(
        #         question, evidence, schema_note, sql_node
        #     )
        # )
        sql_node.warning_cnt += (
            exec_warning_cnt + join_warning_cnt
        )  # + intent_warning_cnt
        if " ||" in sql_node.org_sql:
            sql_node.warning_cnt += 100
        sql_node.notes = {
            "exec_note": exec_note,
            "join_note": join_note,
            # "intent_note": intent_note,
        }
    # # collectively generate the notes for returned columns
    # returned_columns_notes = llm_check(returned_columns_prompts)
    # if evidence is not None and len(evidence.strip()) > 0:
    #     evidence_alignment_notes = llm_check(evidence_alignment_prompts)
    # else:
    #     evidence_alignment_notes = [None] * len(sql_nodes)
    # filtering_conditions_notes = llm_check(filtering_conditions_prompts)
    # for (
    #     sql_node,
    #     # returned_columns_note,
    #     evidence_alignment_note,
    #     # filtering_conditions_note,
    # ) in zip(
    #     sql_nodes,
    #     # returned_columns_notes,
    #     evidence_alignment_notes,
    #     # filtering_conditions_notes,
    # ):
    #     # sql_node.notes["returned_columns_note"] = returned_columns_note
    #     if evidence_alignment_note is not None:
    #         sql_node.notes["evidence_alignment_note"] = evidence_alignment_note
    #         sql_node.evidence_alignment_score = update_evidence_alignment_score(
    #             evidence_alignment_note
    #         )
    #     if sql_node.evidence_alignment_score > 0 and sql_node.warning_cnt < 100:
    #         filtred_sql_nodes.append(sql_node)
    #     # sql_node.notes["filtering_conditions_note"] = filtering_conditions_note
    for sql_node in sql_nodes:
        if sql_node.warning_cnt < 100:
            filtred_sql_nodes.append(sql_node)
    if len(filtred_sql_nodes) == 0:
        return sql_nodes
    else:
        return filtred_sql_nodes


def group_sql_nodes(
    sql_nodes: List[SQLNode], sql_cnt: Dict[str, int], filtering: bool = True
) -> Tuple[List[List[SQLNode]], List[int]]:
    grouped_sql_nodes = defaultdict(list)
    for sql_node in sql_nodes:
        grouped_sql_nodes[frozenset(sql_node.exec_res)].append(sql_node)
    # keep only the top 3 groups of sqls with the same execution results
    group_cnt = defaultdict(int)
    for group, gp_sql_nodes in grouped_sql_nodes.items():
        for sql_node in gp_sql_nodes:
            group_cnt[group] += sql_cnt.get(sql_node.org_sql, 1)
    sorted_group_cnt = sorted(group_cnt.items(), key=lambda x: x[1], reverse=True)
    cnts = [cnt for group, cnt in sorted_group_cnt]
    print(f"[DEBUG][cnts]: {cnts}")
    threshold = cnts[2] if len(cnts) > 2 and filtering else 0
    # threshold = max(sorted_group_cnt[0][1] // 2, threshold)
    print(f"[DEBUG][threshold]: {threshold}")
    filtered_grouped_sql_nodes = []
    filtered_group_cnt = []
    for group, cnt in sorted_group_cnt:
        if cnt < threshold:
            break
        filtered_grouped_sql_nodes.append(grouped_sql_nodes[group])
        filtered_group_cnt.append(cnt)
    print(
        f"[DEBUG][Group Filtering]: {len(grouped_sql_nodes)} -> {len(filtered_grouped_sql_nodes)}"
    )
    return filtered_grouped_sql_nodes, filtered_group_cnt


def intra_group_selection(
    sql_collection: SQLCollection,
    question: str,
    evidence: str,
    grouped_sql_nodes: List[List[SQLNode]],
) -> List[SQLNode]:
    """
    Select the best sql from each group
    """
    intra_group_selected_sqls = []
    prompts = []
    group_sizes = []
    schema_note = collect_schema_info(sql_collection.table_columns, sql_collection.db)
    grouped_sql_nodes_tobe_selected = []
    for gp_sql_nodes in grouped_sql_nodes:
        if len(gp_sql_nodes) == 1:
            intra_group_selected_sqls.append(gp_sql_nodes[0])
            continue
        prompt = intra_group_selection_prompt(
            question, evidence, schema_note, gp_sql_nodes
        )
        prompts.append(prompt)
        group_sizes.append(len(gp_sql_nodes))
        grouped_sql_nodes_tobe_selected.append(gp_sql_nodes)
    # print(f"[DEBUG][prompts]: {prompts[0]}")
    if len(prompts) == 0:
        return intra_group_selected_sqls
    llm_responses = llm_check(prompts)
    selected_ids = []
    for llm_response in llm_responses:
        response = parse_json(llm_response)
        try:
            selected_id = response["best_sql"]
            selected_ids.append(int(selected_id.replace("SQL", "")) - 1)
        except:
            selected_ids.append(0)
    print(f"[DEBUG][selected_ids]: {selected_ids}")
    print(f"[DEBUG][size of each group]: {group_sizes}")
    for gp, selected_id in enumerate(selected_ids):
        selected_id = selected_id % group_sizes[gp]
        intra_group_selected_sqls.append(
            grouped_sql_nodes_tobe_selected[gp][selected_id]
        )
    return intra_group_selected_sqls


def inter_group_selection(
    sql_collection: SQLCollection,
    question: str,
    evidence: str,
    intra_group_selected_sqls: List[SQLNode],
    rules: List[str],
    rule_mode: str,
) -> Tuple[SQLNode, str]:
    """
    Select the best sql from the intra group selected sqls
    """
    if len(intra_group_selected_sqls) == 1:
        return {intra_group_selected_sqls[0]: 1}, []
    base_info = f"""**Schema**
{collect_schema_info(sql_collection.table_columns, sql_collection.db)}

**NL Query**
{question}
**Evidence**
{evidence}
"""
    comparison_prompts = []
    pairs = []
    sql_node_votes = defaultdict(float)
    for i, sql_node1 in enumerate(intra_group_selected_sqls):
        for j, sql_node2 in enumerate(intra_group_selected_sqls):
            if i == j:
                continue
            # if_same_schema = compare_tb_cols(sql_node1, sql_node2)
            # if not if_same_schema:
            #     router_type = "other"
            #     router = compare_other
            if frozenset(sql_node1.exec_res) == frozenset(sql_node2.exec_res):
                sql_node_votes[sql_node1] += 0.5
                sql_node_votes[sql_node2] += 0.5
                continue
            cover_flag = False
            if len(sql_node1.exec_res) == 0 or len(sql_node2.exec_res) == 0:
                pairs.append((sql_node1, sql_node2))
                comparison_prompt = get_prompt(
                    base_info, sql_node1, sql_node2, rule_mode, rules
                )
                comparison_prompts.append(comparison_prompt)
                continue
            if len(sql_node1.exec_res[0]) > len(sql_node2.exec_res[0]):
                cover_flag = if_align_exec_res_with_gt(
                    sql_node1.exec_res, sql_node2.exec_res
                )
                if cover_flag:
                    sql_node_votes[sql_node1] += 1
                    sql_node_votes[sql_node2] += 0
            elif len(sql_node1.exec_res[0]) < len(sql_node2.exec_res[0]):
                cover_flag = if_align_exec_res_with_gt(
                    sql_node2.exec_res, sql_node1.exec_res
                )
                if cover_flag:
                    sql_node_votes[sql_node2] += 1
                    sql_node_votes[sql_node1] += 0
            if cover_flag:
                continue
            pairs.append((sql_node1, sql_node2))
            comparison_prompt = get_prompt(
                base_info, sql_node1, sql_node2, rule_mode, rules
            )
            comparison_prompts.append(comparison_prompt)
    comparison_notes = []
    if len(comparison_prompts) == 0:
        return sql_node_votes, comparison_notes
    llm_responses = llm_check(comparison_prompts)
    for i, (sql_node1, sql_node2) in enumerate(pairs):
        llm_response = llm_responses[i]
        response = parse_json(llm_response)
        if type(response) == dict and "better_sql" in response:
            better_sql = response["better_sql"]
            if better_sql == "SQL1":
                sql_node_votes[sql_node1] += 1
                sql_node_votes[sql_node2] += 0
            elif better_sql == "SQL2":
                sql_node_votes[sql_node2] += 1
                sql_node_votes[sql_node1] += 0
            else:
                sql_node_votes[sql_node1] += 0.5
                sql_node_votes[sql_node2] += 0.5
        else:
            sql_node_votes[sql_node1] += 0.5
            sql_node_votes[sql_node2] += 0.5
        comparison_notes.append(
            {
                "SQL1": sql_node1.org_sql,
                "SQL2": sql_node2.org_sql,
                "comparison_note": llm_response,
            }
        )
        print("-" * 100)
        # print(f"[DEBUG][prompt]: {comparison_prompts[i]}")
        print(f"[DEBUG][comparison_notes]: {comparison_notes[-1]}")
    return sql_node_votes, comparison_notes


def final_adjustment(
    selected_sql_node: SQLNode,
    sql_nodes: List[SQLNode],
) -> SQLNode:
    """
    Find the covered sql
    """
    for sql_node in sql_nodes:
        cover_flag = if_align_exec_res_with_gt(
            sql_node.exec_res, selected_sql_node.exec_res
        )
        if cover_flag:
            selected_sql_node = sql_node
    return selected_sql_node


def update_evidence_alignment_score(evidence_alignment_note: str) -> float:
    try:
        evidence_res = json.loads(evidence_alignment_note)
        evidence_consistency = {}
        for item in evidence_res:
            if item["evidence_claim"] not in evidence_consistency:
                evidence_consistency[item["evidence_claim"]] = 0
            if not item["consistency"].startswith("No"):
                evidence_consistency[item["evidence_claim"]] += 1
        min_consistency = min(evidence_consistency.values())
        if min_consistency == 0:
            return 0
        else:
            return 1
    except:
        return 1


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
Notes on filtering conditions of SQL1:
{sql_node1.notes["filtering_conditions_note"]}

SQL2: {sql_node2.org_sql}
Notes on filtering conditions of SQL2:
{sql_node2.notes["filtering_conditions_note"]}

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
    comparison_note_dict: Dict[str, str],
) -> str:
    # compare the two SQLs with different execution results
    prompt = f"""You are given a natural language query (NL Query) and two SQL statements that produce different results when executed. Your task is to analyze and determine which SQL better aligns with the intent of the NL Query by focusing on their core differences. 

Additional guidelines:
1. Prefer the SQL that returns all the records that satisfy the NL query.
2. If the NL Query asks for the rank of the result, prefer RANK() over ROW_NUMBER() as it is more appropriate for ranking.

**NL query:**
{question}
**Schema information:**
{schema_note}
**Evidence:**
{evidence}

**SQL1**
SQL: {sql_node1.org_sql}
Notes on Execution Results: {sql_node1.notes["exec_note"]}
"""
    if sql_node1.notes["join_note"] is not None:
        prompt += f"Notes on Join Conditions: {sql_node1.notes['join_note']}\n"
    prompt += f"""\n\n**SQL2**
SQL: {sql_node2.org_sql}
Notes on Execution Results: {sql_node2.notes["exec_note"]}
"""
    if sql_node2.notes["join_note"] is not None:
        prompt += f"Notes on Join Conditions: {sql_node2.notes['join_note']}\n"

    prompt += "\n\nCompare the two SQL queries by evaluating them in the following strict priority order:"
    enum_cnt = 1
    if "group_by" in comparison_note_dict:
        prompt += f"\n{enum_cnt}. Group By Clause: {comparison_note_dict['group_by']}"
        enum_cnt += 1
    if "order_by" in comparison_note_dict:
        prompt += f"\n{enum_cnt}. Order By Clause: {comparison_note_dict['order_by']}"
        enum_cnt += 1
    prompt += f"\n{enum_cnt}. Filtering Conditions (e.g., WHERE and HAVING clauses)."
    enum_cnt += 1
    if "col_comparison" in comparison_note_dict:
        prompt += (
            f"\n{enum_cnt}. Returned Columns: {comparison_note_dict['col_comparison']}"
        )
    prompt += """
Please select the better SQL query for the NL query. Your response must be in the following JSON format:
{{
    "reason": "Briefly explain your reasoning process.",
    "better_sql": "SQL1/SQL2/Unsure/Neither",
}}
"""
    return prompt


def multi_aspect_sql_comparison(
    question: str,
    query_type: int,
    evidence: str,
    schema_note: str,
    sql_node1: SQLNode,
    sql_node2: SQLNode,
) -> Tuple[str, str]:
    base_info = f"""
NL Query: {question}
Evidence: {evidence}
Schema: {schema_note}

SQL1: {sql_node1.org_sql}
SQL2: {sql_node2.org_sql}
"""

    # compare if the schema is the same for the two SQLs
    if_same_schema = compare_tb_cols(sql_node1, sql_node2)
    if not if_same_schema:
        router_type = "other"
        router = compare_other
    elif query_type == 1:
        router_type = "top_n"
        router = compare_top_n
    elif query_type == 2:
        router_type = "aggregation"
        router = compare_aggregation
    else:
        router_type = "other"
        router = compare_other
    comparison_prompt = router(base_info, sql_node1, sql_node2)
    # print(comparison_prompt)
    # print("\n" * 5)
    comparison_note_dict = {"router_type": router_type}
    # do the overall comparison
    overall_res = llm_check([comparison_prompt])[0]
    better_sql = parse_result(overall_res)
    comparison_note_dict["overall"] = overall_res
    comparison_note_dict["better_sql"] = better_sql
    return better_sql, comparison_note_dict


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
    selected_sql = sorted_sql_nodes[-1]
    comparison_notes = []
    while True:
        cand_sql_node = None
        for i, sql_node in enumerate(sorted_sql_nodes[::-1]):
            if (
                sql_node not in checked_sql_nodes
                and sql_node.org_sql not in selected_sql.covered_sqls
                and sql_node.org_sql != selected_sql.org_sql
            ):
                cand_sql_node = sql_node
                checked_sql_nodes.append(cand_sql_node)
                break
        if cand_sql_node is None:
            break
        if selected_sql.org_sql in cand_sql_node.covered_sqls:
            selected_sql = cand_sql_node
            continue
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
        if better_sql != "SQL1":
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
    if len(exec_res) != len(gt_exec_res) or len(gt_exec_res) == 0 or len(exec_res) == 0:
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
        # directly skip the sql nodes with evidence alignment score 0
        if sql_node1.evidence_alignment_score == 0:
            continue
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


def rule_based_selection(
    sql_collection: SQLCollection,
    sql_nodes: List[SQLNode],
    question: str,
    evidence: str,
    rules: List[str],
    weights: List[float],
):
    schema_note = collect_schema_info(sql_collection.table_columns, sql_collection.db)
    prompts = []
    for sql_node in sql_nodes:
        for rule in rules:
            prompt = get_rule_evaluation_prompt(
                question, evidence, schema_note, sql_node, rule
            )
            # print(f"[DEBUG][prompt]:\n{prompt}")
            prompts.append(prompt)
    responses = llm_check(prompts)
    max_score = 0
    for i, sql_node in enumerate(sql_nodes):
        score = 0
        for j, rule in enumerate(rules):
            response = responses[i * len(rules) + j]
            answer = parse_json(response)
            if type(answer) != dict or "violation" not in answer:
                score += 0.5 * weights[j]
                sql_node.score_each_rule.append(0.5)
            elif answer["violation"] == "No":
                score += 1 * weights[j]
                sql_node.score_each_rule.append(1)
            elif answer["violation"] == "Yes":
                score += 0
                sql_node.score_each_rule.append(0)
                sql_node.rule_note.append(f"Rule {j}, Reason: {answer['reason']}")
            else:
                score += 0.5 * weights[j]
                sql_node.score_each_rule.append(0.5)
            # if type(answer) != dict or "relevance" not in answer:
            #     sql_node.relevance_each_rule.append("Unsure")
            # else:
            #     sql_node.relevance_each_rule.append(answer["relevance"])
        sql_node.rule_score = score
        max_score = max(max_score, score)
    sql_node_votes = defaultdict(float)
    for sql_node in sql_nodes:
        sql_node_votes[sql_node] = sql_node.rule_score
    return sql_node_votes


def generative_verifier(
    sql_collection: SQLCollection,
    question: str,
    evidence: str,
    sql_nodes: List[SQLNode],
) -> List[SQLNode]:
    schema_note = collect_schema_info(sql_collection.table_columns, sql_collection.db)
    sql_node_votes = defaultdict(float)
    prompts = []
    for sql_node in sql_nodes:
        prompt = get_generative_verifier_prompt(
            question, evidence, schema_note, sql_node
        )
        prompts.append(prompt)
    responses = llm_check(prompts)
    for i, sql_node in enumerate(sql_nodes):
        response = responses[i]
        answer = parse_json(response)
        if type(answer) != dict:
            sql_node_votes[sql_node] = 0.5
        elif answer["correctness"] == "Yes":
            sql_node_votes[sql_node] = 1
        elif answer["correctness"] == "No":
            sql_node_votes[sql_node] = 0
        else:
            sql_node_votes[sql_node] = 0.5
    return sql_node_votes
