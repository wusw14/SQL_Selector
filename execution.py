from typing import Any, Tuple
from analyzer import get_exec_examples
from parser import SQLNode, SubSQLNode
from database import Database
from utils import execute_sql_wrapper, execute_sql


def generate_view_exec_note(view_name: str, view_sql: str, db: Database) -> str:
    # generate the execution note for the view
    sql = f"DROP VIEW IF EXISTS {view_name}"
    exec_res = execute_sql(sql, db.db_path)
    sql = f"CREATE VIEW {view_name} AS {view_sql}"
    exec_res = execute_sql_wrapper(sql, db.db_path, 10)
    sql = f"SELECT * FROM {view_name} limit 3"
    exec_res, exec_columns = execute_sql_wrapper(
        sql, db.db_path, 10, return_columns=True
    )
    if exec_res == "Time Out" or exec_res == "Unexecutable":
        return None
    exec_res_top3 = get_exec_examples(exec_res)
    note = (
        f"The view [{view_name}] contains {len(exec_res)} row(s).\n"
        f"The columns of the view [{view_name}] are: {exec_columns}.\n"
        f"Sample rows (first 3):\n{exec_res_top3}"
    )
    return note


def generate_subsql_exec_note(subsql_node: SubSQLNode, db: Database) -> str:
    sql = subsql_node.sql
    if subsql_node.independent:
        exec_res, exec_columns = execute_sql_wrapper(
            sql, db.db_path, 10, return_columns=True
        )
        exec_res_top3 = get_exec_examples(exec_res)
        note = (
            f"Subquery: {sql}\n"
            f"Execution successful and the number of returned rows is: {len(exec_res)}.\n"
            f"The returned columns are: {exec_columns}.\n"
            f"Sample rows (first 3):\n{exec_res_top3}"
        )
        return note
    else:
        return None


def generate_exec_note(sql_node: SQLNode, db: Database) -> Tuple[str, int]:
    exec_res = sql_node.exec_res
    sql = sql_node.org_sql
    exec_columns = sql_node.exec_columns
    if exec_res == "Time Out":
        note = "[Warning] The execution timed out."
        warning_cnt = 100
    elif exec_res == "Unexecutable":
        note = "[Error] The execution is unexecutable."
        warning_cnt = 100
    elif len(exec_res) == 0 or len(exec_res) == 1 and len(exec_res[0]) == 0:
        note = "[Warning] The execution is successful but the returned rows are empty, which might result from the incorrect filtering conditions or the incorrect join conditions."
        warning_cnt = 10
    elif (
        len(exec_res) == 1
        and len(exec_res[0]) == 1
        and exec_res[0][0] == 0
        and "COUNT" in sql
    ):
        note = "[Warning] The execution is successful but the COUNT result is zero, which might result from the incorrect filtering conditions or the incorrect join conditions."
        warning_cnt = 10
    else:
        exec_res_top3 = get_exec_examples(exec_res)
        note = (
            f"[Info] The SQL query executed successfully and returned {len(exec_res)} row(s).\n"
            f"The returned columns are: {exec_columns}.\n"
            f"Sample rows (first 3):\n{exec_res_top3}"
        )
        # if the sql contains views, generate execution notes for each view name
        if sql_node.views is not None and len(sql_node.views) > 0:
            note += f"\n\nThe views in the SQL are {list(sql_node.views.keys())}."
            note += f"\nThe execution notes for the views are as follows:"
            for view_name, view_sql in sql_node.views.items():
                view_note = generate_view_exec_note(view_name, view_sql, db)
                if view_note is not None:
                    note += f"\n{view_note}"
        if len(sql_node.subsql_nodes) > 1:
            note += f"\n\nFor understanding the SQL query, you can refer to the execution notes of the subqueries as follows:"
            # generate the notes for execution of subqueries
            for subsql_node in sql_node.subsql_nodes[1:]:
                subsql_note = generate_subsql_exec_note(subsql_node, db)
                if subsql_note is not None:
                    note += f"\n{subsql_note}"
        warning_cnt = 0
    return note, warning_cnt
