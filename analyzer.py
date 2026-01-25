from typing import List, Tuple
from parser import SQLNode, SubSQLNode
from database import Database
from typing import Dict
from utils import execute_sql_wrapper
from llm_infer import llm_check
import json
from copy import deepcopy

ANALYSIS_INSTRUCTION = """You are an expert in SQL and natural language understanding. Your task is to convert a given SQL query into a clear, concise, and accurate natural language description (NL). To assist you in this task, the following information is provided:

Schema Information: Details about the database tables, their columns, and relationships.
Execution Results: The output of the SQL query when executed, which can help you understand the query's purpose.
Subquery Analysis: If the SQL query contains subqueries, their purpose and results are provided to help you better interpret the overall query.
Use all the provided information to generate an NL description that fully captures the intent of the SQL query. The description should be concise but detailed enough to recover the SQL query from the NL description.

Guidelines:
1. Clearly describe what the query does, including its filters, groupings, orderings and projections.
2. Mention any specific conditions or values used in the query.
3. Use plain and clear language and avoid technical jargon where possible.
"""


def select_nodes_to_understand(
    sql_nodes: List[SubSQLNode], sql_notes: Dict[SubSQLNode, str]
) -> Tuple[List[SubSQLNode], List[SubSQLNode]]:
    # select the nodes without children or whose children have notes
    nodes_to_understand = []
    for node in sql_nodes:
        if len(node.children) == 0 or all(
            child in sql_notes for child in node.children
        ):
            nodes_to_understand.append(node)
    # remove the selected nodes from the sql_nodes
    sql_nodes = [node for node in sql_nodes if node not in nodes_to_understand]
    # return the selected nodes
    return sql_nodes, nodes_to_understand


def get_exec_examples(result) -> str:
    result_top3 = []
    unique_rows = []
    for row in result:
        if row in unique_rows and len(result) > 3:
            continue
        unique_rows.append(row)
        row_new = []
        for val in row:
            # print(type(val))
            if isinstance(val, str) and len(val) > 100:
                row_new.append(val[:100] + "...")
            elif isinstance(val, list) and len(val) > 100:
                row_new.append(val[:100] + ["..."])
            elif isinstance(val, tuple) and len(val) > 100:
                row_new.append(val[:100] + ("...",))
            else:
                row_new.append(val)
        result_top3.append(f"{row_new}")
        if len(result_top3) >= 3:
            break
    return "\n".join(result_top3)


def generate_note(
    node: SubSQLNode,
    question: str,
    evidence: str,
    db: Database,
    sql_notes: Dict[SubSQLNode, str],
    if_main: bool,
    table_columns: Dict[str, List[str]],
    views_note: str = None,
    exec_note: str = None,
) -> str:
    # Step 1: if independent, execute the sql and get the result
    if node.independent and exec_note is None:
        result = execute_sql_wrapper(node.sql, db.db_path, 10)
        if result == "Time Out" or result == "Unexecutable":
            exec_note = f"Execution failed: {result}"
        elif len(result) == 0 or len(result) == 1 and len(result[0]) == 0:
            exec_note = "[Warning] The execution is successful but the returned rows are empty, which might result from the incorrect filtering conditions."
        elif (
            len(result) == 1
            and len(result[0]) == 1
            and result[0][0] == 0
            and "COUNT" in node.sql
        ):
            exec_note = "[Warning] The execution is successful but the COUNT result is zero, which might result from the incorrect filtering conditions."
        else:
            result = get_exec_examples(result)
            exec_note = f"Execution successful and the number of returned rows is: {len(result)}. The top 3 rows are: {result}"
    # else:
    #     exec_note = None
    # Step 2: collect all the relevant schema information
    table_columns_new = deepcopy(table_columns)
    for tb_name, columns in node.table_columns.items():
        if tb_name in table_columns_new:
            table_columns_new[tb_name] = list(
                set(table_columns_new[tb_name]) | set(columns)
            )
        else:
            table_columns_new[tb_name] = columns
    schema_note = collect_schema_info(table_columns_new, db)
    # Step 3: subqueries notes
    subqueries_note = get_subqueries_note(node, sql_notes)
    # Step 4: form the prompt
    prompt = form_prompt(
        question,
        evidence,
        node.sql,
        exec_note,
        schema_note,
        subqueries_note,
        if_main,
        views_note,
    )
    # print("[DEBUG][prompt]:", prompt)
    # Step 5: generate the note
    note = llm_check([prompt])[0]
    # print(node.sql)
    # print("[DEBUG][note]:", note)
    # print(note)
    return note


def form_prompt(
    question: str,
    evidence: str,
    generated_sql: str,
    exec_note: str,
    schema_note: str,
    subqueries_note: str,
    if_main: bool,
    views_note: str = None,
) -> str:
    if if_main:
        prompt = ANALYSIS_INSTRUCTION
        prompt += "\nFinally, compare the generated NL description of the SQL with the original NL query, and determine whether the SQL query’s intention is consistent with the intention described in the original NL query. If there are discrepancies, highlight the differences and explain why the SQL query does not fully align with the original NL query."
        if views_note is not None and len(views_note.strip()) > 0:
            prompt += f"\n**Views and their notes:**\n{views_note}"
        prompt += f"\n**Generated SQL:**\n{generated_sql}"
        if exec_note is not None:
            prompt += f"\n**Execution result:**\n{exec_note}"
        prompt += f"\n**Relevant schema information:**\n{schema_note}"
        if evidence is not None and len(evidence.strip()) > 0:
            prompt += f"\n**Evidence:**\n{evidence}"
        if subqueries_note is not None:
            prompt += f"\n**Subquery notes:**(For easy understanding, you can refer to the subquery notes to help you analyze the SQL query.)\n{subqueries_note}"
        prompt += f"\n**Original NL query:**\n{question}"
        prompt += """\nWhen responding, ensure your output is formatted as valid JSON with the following structure:
```json
{
  "intention_analysis": "Briefly explain your analysis of the SQL query's intention.",
  "intention": "Summarize the SQL query's intention in one concise sentence.",
  "consistency_analysis": "Briefly explain your analysis of the SQL query's intention consistency with the original NL query.",
  "consistency": "Yes/No/Unsure",
}
```
"""
    else:
        # form the prompt for the subquery
        prompt = ANALYSIS_INSTRUCTION
        if views_note is not None and len(views_note.strip()) > 0:
            prompt += f"\n**Views and their notes:**\n{views_note}"
        prompt += f"\n**Generated SQL:**\n{generated_sql}"
        if exec_note is not None:
            prompt += f"\n**Execution result:**\n{exec_note}"
        prompt += f"\n**Relevant schema information:**\n{schema_note}"
        if evidence is not None and len(evidence.strip()) > 0:
            prompt += f"\n**Evidence:**\n{evidence}"
        if subqueries_note is not None:
            prompt += f"\n**Subquery notes:**(For easy understanding, you can refer to the subquery notes to help you analyze the SQL query.)\n{subqueries_note}"
        prompt += """\nWhen responding, ensure your output is formatted as valid JSON with the following structure:
```json
{
  "intention_analysis": "Briefly explain your analysis of the SQL query's intention.",
  "intention": "Summarize the SQL query's intention in one concise sentence."
}
```
"""
    return prompt


def get_subqueries_note(node: SQLNode, sql_notes: Dict[SQLNode, str]) -> str:
    if len(node.children) == 0:
        return None
    subqueries_notes = []
    for child in node.children:
        if child in sql_notes:
            subqueries_notes.append(
                f"Subquery: {child.sql}\nNote: {sql_notes[child]}\n"
            )
    return "\n".join(subqueries_notes)


def collect_schema_info(table_columns: Dict[str, List[str]], db: Database) -> str:
    schema_info_list = [f"Database: {db.db_name}"]
    for tb_name in table_columns:
        columns = table_columns[tb_name]
        schema_info_list.append(db.tables[tb_name].display(columns))
    schema_info = "\n".join(schema_info_list)
    schema_info = schema_info.replace("\n\n", "\n")
    return schema_info


def classify_query_type(question: str) -> int:
    prompt = f"""Classify the following natural language query into exactly one of these three categories:

Selection of Top/Best/Worst/Most – The query asks to choose the most, least, best, worst, top N, or similar superlative selection from a set.
Example: “Which product has the highest sales?”, “Show the 5 oldest customers.”

Aggregation of Filtered Results – The query asks to apply an aggregation (e.g., sum, count, average, total) to a filtered subset of data.
Example: “What is the total revenue from Europe last year?”, “Count the number of pending orders.”

Other Record Retrieval – The query asks to return all records (or a filtered set) that match given conditions, without aggregation or top/best selection.
Example: “List all customers from Germany.”, “Show orders placed in January.”

**NL query:**
{question}

Respond only with the category number (1, 2, or 3). Do not include explanations.
"""
    res = llm_check([prompt])[0]
    if res.startswith("1"):
        return 1
    elif res.startswith("2"):
        return 2
    elif res.startswith("3"):
        return 3
    else:
        return 3


def understand_sql(
    sql_node: SQLNode,
    question: str,
    evidence: str,
    db: Database,
    table_columns: Dict[str, List[str]],
    exec_note: str,
) -> str:
    # understand sqls hierarchically by generating notes for each node
    sql_notes = {}
    sql_nodes = sql_node.subsql_nodes
    views_note = sql_node.views_note
    root_node = sql_nodes[0]
    while len(sql_nodes) > 0:
        # select the nodes without children or whose children have notes
        sql_nodes, nodes_to_understand = select_nodes_to_understand(
            sql_nodes, sql_notes
        )
        # generate notes for the selected nodes
        # if len(sql_nodes) == 0:
        #     if_main = True
        # else:
        if_main = False
        if len(sql_nodes) != 0:
            exec_note = None
        for node in nodes_to_understand:
            node.note = generate_note(
                node,
                question,
                evidence,
                db,
                sql_notes,
                if_main,
                table_columns,
                views_note,
                exec_note,
            )
            print(node.sql)
            print(node.note)
            print("-" * 100)
        # update the sql_notes
        sql_notes.update({node: node.note for node in nodes_to_understand})
    # check if the SQL aligns with NL query based on the note
    if_align = check_if_align(question, sql_node.org_sql, root_node.note)
    warning_cnt = 0 if if_align else 5
    return root_node.note, warning_cnt


def check_if_align(question: str, sql: str, note: str) -> bool:
    prompt = f"""You are a SQL analysis expert. Your task is to determine whether the SQL query's intention is consistent with the intention described in the NL query. The note for the intention of the SQL query is provided for reference.

**NL query:**
{question}

**SQL**
{sql}

**Note for the intention of the SQL query:**
{note}

Please check if the SQL query's intention is consistent with the intention described in the NL query. Respond with "Yes" or "No" without any other words.
"""
    res = llm_check([prompt])[0]
    if res == "Yes":
        return True
    else:
        return False
