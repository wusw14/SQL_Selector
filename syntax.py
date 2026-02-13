from typing import Any, Tuple, List
from analyzer import get_exec_examples
from parser import SQLNode, prepare_attr_map_dict, recover_sql_from_attr_map_dict
from database import Database
import re
from sqlglot import parse_one, exp


def extract_group_by(sql: str) -> Tuple[List[str], str]:
    pattern = "(?i)GROUP\s+BY\s+((?:(?!\b(?:ORDER\s+BY|LIMIT|HAVING)\b)[^)(])+)"
    match = re.search(pattern, sql, re.IGNORECASE | re.DOTALL)
    if match:
        clause = match.group(1)
        cols = []
        for col in clause.split(","):
            col = col.strip()
            if "." in col:
                tb, col = col.strip().split(".", maxsplit=1)
            cols.append(col)
        return cols, clause
    return [], None


def classify_ranking(sql: str) -> str:
    if (
        "ORDER BY" not in sql.upper()
        and "MAX" not in sql.upper()
        and "MIN" not in sql.upper()
    ):
        return None
    if "ORDER BY" in sql.upper():
        if "DESC" in sql.upper():
            return "order by desc"
        else:
            return "order by asc"
    elif "MAX" in sql.upper():
        return "max"
    elif "MIN" in sql.upper():
        return "min"
    return None


def extract_join_conditions(sql: str) -> str:
    # parse the sql to get the join conditions
    attr_map_dict, sql = prepare_attr_map_dict(sql)
    try:
        parsed = parse_one(sql)
    except Exception as e:
        return []
    # table alias map
    table_names = set()
    table_alias_map = {}
    for table in parsed.find_all(exp.Table):
        table_alias_map[table.alias] = table.name
        table_names.add(table.name)
        table_names.add(table.alias)
    joins = parsed.find_all(exp.Join)
    join_conditions = []
    # INNER JOIN frpm AS T2 ON T1.CDSCode = T2.CDSCode
    for join in joins:
        join_sql = join.sql()
        join_sql = recover_sql_from_attr_map_dict(join_sql, attr_map_dict)
        # parse the tables and columns from the join sql
        if " ON " not in join_sql or "=" not in join_sql:
            continue
        join_condition = join_sql.split(" ON ", maxsplit=1)[-1]
        if "=" not in join_condition:
            continue
        left, right = join_condition.split("=", maxsplit=1)
        if "." not in left or "." not in right:
            continue
        ltb, lcol = left.strip().split(".", maxsplit=1)
        lcol = lcol.strip().split(" AND ")[0]
        rtb, rcol = right.strip().split(".", maxsplit=1)
        rcol = rcol.strip().split(" AND ")[0]
        ltb = table_alias_map.get(ltb, ltb)
        rtb = table_alias_map.get(rtb, rtb)
        join_conditions.append((ltb, lcol, rtb, rcol))
    # where conditions
    where_conditions = parsed.find_all(exp.Where)
    where_conditions = [where.sql() for where in where_conditions]
    # WHERE T1.FundingType = 'Locally funded' AND T1.district = T2.district AND T1.score > (SELECT avg_diff FROM AverageDifference)
    for where_condition in where_conditions:
        where_sql = where_condition[5:].strip()
        where_sql = recover_sql_from_attr_map_dict(where_sql, attr_map_dict)
        conditions = where_sql.split(" AND ")
        for condition in conditions:
            if "=" not in condition:
                continue
            left, right = condition.split("=", maxsplit=1)
            if "." not in left or "." not in right:
                continue
            ltb, lcol = left.strip().split(".", maxsplit=1)
            rtb, rcol = right.strip().split(".", maxsplit=1)
            ltb = table_alias_map.get(ltb, ltb)
            rtb = table_alias_map.get(rtb, rtb)
            if ltb not in table_names or rtb not in table_names:
                continue
            join_conditions.append((ltb, lcol, rtb, rcol))
    return join_conditions


def generate_join_note(sql_node: SQLNode, db: Database) -> Tuple[str, int]:
    # extract the join conditions for views and SQLs
    join_conditions_all = []
    for view_name, view_sql in sql_node.views.items():
        join_conditions = extract_join_conditions(view_sql)
        for cond in join_conditions:
            if cond not in join_conditions_all:
                join_conditions_all.append(cond)
    join_conditions = extract_join_conditions(sql_node.main_sql_node.sql)
    for cond in join_conditions:
        if cond not in join_conditions_all:
            join_conditions_all.append(cond)
    note = ""
    warning_cnt = 0
    joinable_columns = db.joinable_columns
    # TODO: collect the incorrect join conditions
    incorrect_join_conditions = []
    non_unique_join_conditions = []
    for ltb, lcol, rtb, rcol in join_conditions_all:
        ltb = ltb.replace("`", "")
        rtb = rtb.replace("`", "")
        ltb = ltb.lower()
        rtb = rtb.lower()
        if ltb == rtb:
            continue
        lcol = lcol.lower().replace('"', "").replace("`", "")
        rcol = rcol.lower().replace('"', "").replace("`", "")
        if ltb not in db.tables or rtb not in db.tables:
            continue
        if lcol not in db.tables[ltb].columns or rcol not in db.tables[rtb].columns:
            continue
        # print(f"[DEBUG] ltb: {ltb}, lcol: {lcol}, rtb: {rtb}, rcol: {rcol}")
        lcol_obj = db.tables[ltb].columns[lcol]
        rcol_obj = db.tables[rtb].columns[rcol]
        if not lcol_obj.if_unique and not rcol_obj.if_unique:
            non_unique_join_conditions.append(f"{ltb}.{lcol} = {rtb}.{rcol}")
        if (ltb, lcol) in joinable_columns:
            tb_cols = joinable_columns[(ltb, lcol)]
            if (rtb, rcol) not in tb_cols:
                incorrect_join_conditions.append((ltb, lcol, rtb, rcol))
    if len(non_unique_join_conditions) > 0:
        note += f"[Warning] For the join conditions {non_unique_join_conditions}, both columns contain non-unique values, which might result in duplicated rows when joining the tables."
        warning_cnt += 5
    if len(incorrect_join_conditions) > 0:
        note += f"[Error] For the join conditions {incorrect_join_conditions}, the columns are not joinable, which might result in incorrect results."
        warning_cnt += 100
    return note, warning_cnt


def extract_group_by_regex(sql):
    match = re.search(
        r"GROUP\s+BY\s+([^;]+?)(?:\s+(?:ORDER|HAVING|LIMIT|;))",
        sql,
        re.IGNORECASE | re.DOTALL,
    )
    if match:
        clause = match.group(1)
        return [col.strip() for col in clause.split(",")]
    return []


def extract_select_conditions(sql):
    match = re.search(
        r"SELECT\s+([^;]+?)(?:\s+(?:FROM|WHERE|GROUP|ORDER|HAVING|LIMIT|;))",
        sql,
        re.IGNORECASE | re.DOTALL,
    )
    if match:
        clause = match.group(1)
        return [col.strip() for col in clause.split(",")]
    return []


def compare_tb_cols(sql_node1: SQLNode, sql_node2: SQLNode) -> bool:
    sql1_tbs = sql_node1.aligned_tables
    sql2_tbs = sql_node2.aligned_tables
    if frozenset(sql1_tbs) != frozenset(sql2_tbs):
        return False
    sql1_cols = sql_node1.aligned_columns
    sql2_cols = sql_node2.aligned_columns
    if frozenset(sql1_cols) != frozenset(sql2_cols):
        return False
    return True


# def generate_group_note(sql_node: SQLNode) -> Tuple[str, int]:
#     subsql_nodes = sql_node.subsql_nodes
#     nodes_with_group_by = []
#     # check the node that has group by conditions
#     for node in subsql_nodes:
#         if "GROUP BY" in node.sql:
#             flag = True
#             for child in node.children:
#                 if "GROUP BY" in child.sql:
#                     flag = False
#                     break
#             if flag:
#                 nodes_with_group_by.append(node)

#     note = ""
#     warning_cnt = 0
#     # check the group by conditions
#     for node in nodes_with_group_by:
#         group_by_conditions = extract_group_by_regex(node.sql)
#         select_conditions = extract_select_conditions(node.sql)
#         abnormal_select_conditions = []
#         for select_condition in select_conditions:
#             if "(" in select_condition:
#                 continue
#             if select_condition not in group_by_conditions:
#                 abnormal_select_conditions.append(select_condition)
#         if len(abnormal_select_conditions) > 0:
#             note += f"[Warning] The select attributes {abnormal_select_conditions} are not in the group by conditions, which might result in incorrect results."
#             warning_cnt += 1
#     return note, warning_cnt
