from sqlglot import parse_one, exp
from typing import List, Dict, Tuple, Set, Any
import re
from database import Database
from sql_metadata import Parser
from collections import defaultdict
from utils import execute_sql_wrapper
import time


def collect_schema_info(table_columns: Dict[str, List[str]], db: Database) -> str:
    schema_info_list = [f"Database: {db.db_name}"]
    for tb_name in table_columns:
        columns = table_columns[tb_name]
        schema_info_list.append(db.tables[tb_name].display(columns))
    schema_info = "\n".join(schema_info_list)
    schema_info = schema_info.replace("\n\n", "\n")
    return schema_info


class SubSQLNode:
    def __init__(self, sql: str, parent: "SubSQLNode" = None):
        self.sql = self.normalize_sql(sql)
        self.parent = parent
        self.columns = []
        self.tables = []
        self.independent = None
        self.children = []
        self.note = None
        self.table_columns = defaultdict(list)

    def normalize_sql(self, sql: str) -> str:
        # remove the comments
        sql = re.sub(r"--.*", "", sql)
        sql = re.sub(r"/\*.*\*/", "", sql)
        sql = sql.strip()
        # if sql starts with "(" and ends with ")", remove them
        if sql.startswith("("):
            # find the matching ")" from right to left
            index = sql.rfind(")")
            if index != -1:
                sql = sql[1:index]
            else:
                sql = sql[1:]
        return sql

    def update_tables_and_columns(self):
        try:
            parser = Parser(self.sql)
            self.tables = set([tb.lower() for tb in parser.tables])
            self.columns = set([col.lower() for col in parser.columns])
        except Exception as e:
            print("SQL is not valid: ", self.sql)

    def update_if_independent(self, db: Database):
        # check if each column is in the table
        for col_name in self.columns:
            flag = False
            if "." in col_name:
                tb_name, col_name = col_name.split(".", maxsplit=1)
                if tb_name in db.tables and col_name in db.tables[tb_name].columns:
                    flag = True
                    self.table_columns[tb_name].append(col_name)
                else:
                    self.independent = False
                    break
            else:
                for tb_name in self.tables:
                    if tb_name in db.tables and col_name in db.tables[tb_name].columns:
                        flag = True
                        self.table_columns[tb_name].append(col_name)
                        break
            if not flag and self.independent is None:
                self.independent = False
                break
        if self.independent is None:
            self.independent = True


class SQLNode:
    def __init__(
        self,
        org_sql: str,
        subsql_nodes: List[SubSQLNode],
        views: Dict[str, str],
        exec_res: Any,
        exec_time: float,
        exec_columns: List[str],
        tables: List[str],
        columns: List[str],
    ):
        self.org_sql = org_sql
        self.main_sql_node = subsql_nodes[0]
        self.subsql_nodes = subsql_nodes
        self.views = views
        self.exec_res = exec_res
        self.exec_time = exec_time
        self.tables, self.columns = tables, columns
        self.aligned_tables, self.aligned_columns = [], []
        self.warning_cnt = 0
        self.notes = None
        self.views_note = None
        self.rank = None
        self.exec_columns = exec_columns
        self.covered_sqls = []
        self.coverage = 0
        self.evidence_alignment_score = 1
        self.acc = None
        self.rule_score = 0
        self.score_each_rule = []
        self.relevance_each_rule = []
        self.rule_note = []

    def find_table_columns(self) -> Tuple[Set[str], Set[str]]:
        tables = set()
        columns = set()
        main_sql = self.main_sql_node.sql
        try:
            parser = Parser(main_sql)
            tables.update(parser.tables)
            columns.update(parser.columns)
        except Exception as e:
            print(f"[Error] Failed to parse the main SQL: {e}")
            print(f"[SQL]: {main_sql}")
        for view_name, view_sql in self.views.items():
            try:
                parser = Parser(view_sql)
                tables.update(parser.tables)
                columns.update(parser.columns)
            except Exception as e:
                print(f"[Error] Failed to parse the view SQL: {e}")
                print(f"[SQL]: {view_sql}")
        tables = [tb.lower() for tb in tables]
        columns = [col.lower() for col in columns]
        return tables, columns

    def align_tables_and_columns(self, db: Database):
        aligned_tables = []
        aligned_columns = []
        for table in self.tables:
            if table in db.tables:
                aligned_tables.append(table)
        for col_name in self.columns:
            if "." in col_name:
                tb_name, col_name = col_name.split(".", maxsplit=1)
                if (
                    tb_name in aligned_tables
                    and col_name in db.tables[tb_name].columns
                    and col_name not in aligned_columns
                ):
                    aligned_columns.append(col_name)
            else:
                for tb_name in aligned_tables:
                    if (
                        col_name in db.tables[tb_name].columns
                        and col_name not in aligned_columns
                    ):
                        aligned_columns.append(col_name)
                        break
        self.aligned_tables = aligned_tables
        self.aligned_columns = aligned_columns


class SQLCollection:
    def __init__(self, sqls: List[str], db: Database, info: Dict[str, Any]):
        self.db = db
        self.exe_results, self.sql_times, self.sql_columns = self.get_exe_results(sqls)
        self.sqls, self.sql_nodes = self.parse_sqls(sqls)
        # self.prefilter_by_tables()
        # TODO: self.prefilter_by_conditions()
        self.tables, self.columns = self.agg_table_columns()
        self.table_columns = self.find_table_columns()
        self.intra_selected_sql_nodes = []
        self.rules = []
        self.filtered_rules = []
        self.rule_scores = {}
        self.schema_note = collect_schema_info(self.table_columns, self.db)
        self.info = info
        self.gt_sql_nodes = None  # the SQL nodes for the ground truth
        self.incorrect_sql_nodes = None
        self.comparison_notes = []

    def get_exe_results(self, sqls: List[str]) -> Dict[str, str]:
        """
        Get the execution results of the sqls
        Only keep the sqls that are executable
        Return: a dictionary of sql -> execution result
        """
        sql_results = {}
        sql_times = {}
        sql_columns = {}
        for sql in sqls:
            if "FROM" not in sql.upper():
                continue
            start_time = time.time()
            res, cols = execute_sql_wrapper(
                sql, self.db.db_path, 10, return_columns=True
            )
            time_cost = time.time() - start_time
            if res == "Time Out" or res == "Unexecutable":
                continue
            # if (
            #     len(res) == 0
            #     or len(res) == 1
            #     and (len(res[0]) == 0 or len(res[0]) == 1 and res[0][0] is None)
            # ):
            #     continue
            sql_results[sql] = res
            sql_times[sql] = time_cost
            sql_columns[sql] = cols
        return sql_results, sql_times, sql_columns

    def parse_sqls(self, sqls: List[str]) -> Tuple[List[str], List[SQLNode]]:
        filtered_sqls, sql_nodes = [], []
        for sql in sqls:
            if sql not in self.exe_results:
                continue
            subsql_nodes, views, tables, columns = parse_sql(sql)
            tables = [t.lower() for t in tables]
            columns = [col.lower() for col in columns]
            exec_res = self.exe_results[sql]
            exec_time = self.sql_times[sql]
            exec_columns = self.sql_columns[sql]
            if tables is None or columns is None:
                continue
            sql_node = SQLNode(
                sql,
                subsql_nodes,
                views,
                exec_res,
                exec_time,
                exec_columns,
                tables,
                columns,
            )
            filtered_sqls.append(sql)
            sql_node.align_tables_and_columns(self.db)
            sql_nodes.append(sql_node)
        return filtered_sqls, sql_nodes

    def agg_table_columns(self) -> Tuple[Set[str], Set[str]]:
        tables = set()
        columns = set()
        for sql_node in self.sql_nodes:
            tables.update(sql_node.tables)
            columns.update(sql_node.columns)
        return tables, columns

    def find_table_columns(self) -> Dict[str, List[str]]:
        table_columns = defaultdict(list)
        # check if each column is in the table
        for col_name in self.columns:
            if "." in col_name:
                tb_name, col_name = col_name.split(".", maxsplit=1)
                if (
                    tb_name in self.db.tables
                    and col_name in self.db.tables[tb_name].columns
                    and col_name not in table_columns[tb_name]
                ):
                    table_columns[tb_name].append(col_name)
            else:
                for tb_name in self.tables:
                    if (
                        tb_name in self.db.tables
                        and col_name in self.db.tables[tb_name].columns
                        and col_name not in table_columns[tb_name]
                    ):
                        table_columns[tb_name].append(col_name)
        for tb in self.tables:
            if tb not in table_columns and tb in self.db.tables:
                table_columns[tb].extend(self.db.tables[tb].primary_keys)
        return table_columns

    def prefilter_by_tables(self):
        # filter the SQL with the tables via majority voting strategy
        tables_sqlnodes = defaultdict(list)
        for sql_node in self.sql_nodes:
            tables = sql_node.tables
            tables = [tb for tb in tables if tb in self.db.tables]
            tables = frozenset(tables)
            tables_sqlnodes[tables].append(sql_node)
        max_vote = max(len(sqlnodes) for sqlnodes in tables_sqlnodes.values())
        filtered_sqls, filtered_sql_nodes = [], []
        for tables, sqlnodes in tables_sqlnodes.items():
            if len(sqlnodes) == max_vote:
                filtered_sqls.extend([sql_node.org_sql for sql_node in sqlnodes])
                filtered_sql_nodes.extend(sqlnodes)
        self.sqls = filtered_sqls
        self.sql_nodes = filtered_sql_nodes


def prepare_attr_map_dict(sql: str):
    """
    Prepare the attribute map dictionary for the SQL
    Target: to deal with the attribute names with special characters
    Find all the names in `XXX`
    """
    attr_map_dict = {}
    attr_names = set()
    for match in re.findall(r"`([^`]+)`", sql):
        attr_names.add(match)
    # replace the "+-*/()[] " with "_"
    for attr_name in attr_names:
        attr_new = re.sub(r"[+\-*/%()\[\] ]", "", attr_name)
        attr_map_dict[attr_new] = f"`{attr_name}`"
        sql = sql.replace(f"`{attr_name}`", attr_new)
    return attr_map_dict, sql


def recover_sql_from_attr_map_dict(sql: str, attr_map_dict: Dict[str, str]):
    """
    Recover the sql from the attribute map dictionary
    """
    for attr_new, attr_name in attr_map_dict.items():
        sql = sql.replace(attr_new, attr_name)
    return sql


def parse_sql(sql: str) -> Tuple[List[SubSQLNode], Dict[str, str]]:
    """
    Parse the sql into a tree structure
    Child nodes are the subqueries of the SQL
    """
    # normalize the sql by removing the comments
    sql = re.sub(r"--.*", "", sql)
    sql = re.sub(r"/\*.*\*/", "", sql)
    sql = sql.strip()
    # for easy parsing, modify the attribute names
    # attr_map_dict: attr_new -> attr_name
    attr_map_dict, sql = prepare_attr_map_dict(sql)
    # remove the redundant spaces
    sql = re.sub(r"\s+", " ", sql)
    sql = sql.strip()
    views = {}
    try:
        parsed_query = parse_one(sql)
        tables = parsed_query.find_all(exp.Table)
        # tables = list(set([table.name.lower() for table in tables]))
        tables = list(set([table.name for table in tables]))
        columns = parsed_query.find_all(exp.Column)
        # columns = list(set([column.name.lower() for column in columns]))
        columns = list(set([column.name for column in columns]))
        tables = [
            recover_sql_from_attr_map_dict(table, attr_map_dict) for table in tables
        ]
        columns = [
            recover_sql_from_attr_map_dict(column, attr_map_dict) for column in columns
        ]
        ctes = parsed_query.find_all(exp.CTE)
        views = {cte.alias: cte.this.sql() for cte in ctes}
        parsed_query.set("with", None)
        sql = parsed_query.sql()
        subqueries = parsed_query.find_all(exp.Subquery)
        subqueries_sql = [subquery.sql() for subquery in subqueries]
        subqueries_sql = [
            recover_sql_from_attr_map_dict(q, attr_map_dict) for q in subqueries_sql
        ]
        views = {
            k: recover_sql_from_attr_map_dict(v, attr_map_dict)
            for k, v in views.items()
        }
    except Exception as e:
        print(f"[Error] Failed to parse the SQL: {e}")
        print(f"[SQL]: {sql}")
        subqueries_sql = None
        tables = None
        columns = None
    sql = recover_sql_from_attr_map_dict(sql, attr_map_dict)
    # sort the subqueries_sql by the length of the sql
    sql_nodes = [SubSQLNode(sql)]
    if subqueries_sql is None:
        return sql_nodes, views, tables, columns
    subqueries_sql.sort(key=lambda x: len(x))
    for q in subqueries_sql:
        q_node = SubSQLNode(q, sql_nodes[0])
        sql_nodes.append(q_node)
    # find the parent for each subquery
    for i in range(1, len(sql_nodes)):
        for j in range(i + 1, len(sql_nodes)):
            if sql_nodes[i].sql in sql_nodes[j].sql:
                sql_nodes[i].parent = sql_nodes[j]
                break
    # find the children for each subquery
    for i in range(len(sql_nodes)):
        for j in range(len(sql_nodes)):
            if sql_nodes[i].parent == sql_nodes[j]:
                sql_nodes[j].children.append(sql_nodes[i])
    return sql_nodes, views, tables, columns
