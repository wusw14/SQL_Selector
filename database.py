from typing import List, Dict, Any, Tuple
from utils import execute_sql_wrapper
import numpy as np
import os
import pandas as pd
from collections import defaultdict
from llm_infer import llm_check


class Column:
    def __init__(
        self,
        db_path: str,
        tb_name: str,
        name: str,
        desp: str = None,
    ):
        self.db_path = db_path
        self.tb_name = tb_name
        self.name = name
        self.desp = desp
        self.if_unique = False
        self.stats = self.get_stats()
        self.val_examples = self.get_val_examples()

    def update_desp(self, desp: str):
        self.desp = desp

    def get_val_examples(self):
        sql = f"SELECT DISTINCT `{self.name}` FROM `{self.tb_name}` WHERE `{self.name}` IS NOT NULL AND `{self.name}` != ''"
        if self.stats["number of distinct values"] > 20:
            sql += " LIMIT 3;"
        else:
            sql += ";"
        res = execute_sql_wrapper(sql, self.db_path, 10)
        res = [row[0] for row in res]
        return res

    def get_stats(self):
        # num of distinct values, ratio of null values
        sql = f"SELECT COUNT(DISTINCT `{self.name}`) FROM `{self.tb_name}`;"
        num_distinct = execute_sql_wrapper(sql, self.db_path, 10)
        if num_distinct is None:
            num_distinct = 0
        else:
            try:
                num_distinct = int(num_distinct[0][0])
            except:
                print(f"[Error] {num_distinct} is not a number")
                num_distinct = 0
        sql = f"SELECT cast(COUNT(`{self.name}`) as real) / cast(COUNT(*) as real), COUNT(`{self.name}`) FROM `{self.tb_name}`;"
        res = execute_sql_wrapper(sql, self.db_path, 10)
        try:
            null_ratio = np.round(100 - float(res[0][0]) * 100, 2)
            num_values = int(res[0][1])
            if num_values == num_distinct:
                self.if_unique = True
        except:
            null_ratio = 100
            num_values = 0
        return {
            "number of distinct values": num_distinct,
            "ratio of null values(%)": null_ratio,
            "duplication": not self.if_unique,
        }

    def display(self) -> str:
        meta_data = {
            "description": self.desp,
            "example values": self.val_examples,
        }
        for key, value in self.stats.items():
            meta_data[key] = value
        output = f"{self.name}: {meta_data}\n"
        return output


class Table:
    def __init__(
        self,
        db_path: str,
        db_name: str,
        tb_name: str,
        tb_desp: pd.DataFrame = None,
    ):
        self.db_path = db_path
        self.db_name = db_name
        self.tb_name = tb_name
        self.primary_keys = []
        self.joinable_columns = defaultdict(list)
        self.columns = self.load_columns(tb_desp)

    def load_columns(self, tb_desp: pd.DataFrame = None) -> Dict[str, Column]:
        column_dict = {}
        column_desp = {}
        # primary key
        sql = f"PRAGMA table_info(`{self.tb_name}`);"
        res = execute_sql_wrapper(sql, self.db_path, 10)
        for col_info in res:
            col_name = col_info[1].lower()
            primary_key = col_info[-1]
            if primary_key > 0:
                desp = "Primary key"
                self.primary_keys.append(col_name)
            else:
                desp = None
            column_desp[col_name] = desp
        # foreign key
        sql = f"PRAGMA foreign_key_list(`{self.tb_name}`);"
        res = execute_sql_wrapper(sql, self.db_path, 10)
        for col_info in res:
            parent_table = col_info[2].lower() if col_info[2] is not None else None
            child_col = col_info[3].lower() if col_info[3] is not None else None
            parent_col = col_info[4].lower() if col_info[4] is not None else None
            if parent_col is None:
                continue
            desp = f"Foreign key to {parent_table}.{parent_col}"
            self.joinable_columns[(self.tb_name, child_col)].append(
                (parent_table, parent_col)
            )
            if column_desp[child_col] is not None:
                column_desp[child_col] = f"{column_desp[child_col]} | {desp}"
            else:
                column_desp[child_col] = desp

        if tb_desp is not None:
            for _, row in tb_desp.iterrows():
                col_name = row["original_column_name"].strip().lower()
                desp_list = []
                if (
                    type(row["column_name"]) == str
                    and len(row["column_name"].strip()) > 0
                ):
                    desp_list.append(f"{row['column_name'].strip()}")
                if (
                    type(row["column_description"]) == str
                    and len(row["column_description"].strip()) > 0
                ):
                    desp_list.append(f"{row['column_description'].strip()}")
                if (
                    type(row["value_description"]) == str
                    and len(row["value_description"].strip()) > 0
                ):
                    desp_list.append(f"{row['value_description'].strip()}")
                desp = " | ".join(desp_list)
                if len(desp.strip()) > 0:
                    if column_desp[col_name] is not None:
                        column_desp[col_name] = (
                            f"{column_desp[col_name]} | {desp.strip()}"
                        )
                    else:
                        column_desp[col_name] = desp.strip()
        for col_name, desp in column_desp.items():
            col = Column(self.db_path, self.tb_name, col_name, desp)
            column_dict[col_name] = col
        return column_dict

    def display(self, col_names: List[str] = None):
        if col_names is None:
            col_names = list(self.columns.keys())
        output = f"Table: {self.tb_name}"
        # number of records
        sql = f"SELECT COUNT(*) FROM `{self.tb_name}`;"
        num_records = execute_sql_wrapper(sql, self.db_path, 10)
        if num_records is None:
            num_records = 0
        else:
            num_records = num_records[0][0]
        output += f" (Number of records: {num_records})\n"
        for col_name in col_names:
            if col_name not in self.columns:
                continue
            output += self.columns[col_name].display()
        output += "\n"
        return output


def update_joinable_columns(
    joinable_columns: Dict[Tuple[str, str], List[Tuple[str, str]]],
) -> Dict[Tuple[str, str], List[Tuple[str, str]]]:
    agg_joinable_columns = {}
    for (tb_name, col_name), joinable_cols in joinable_columns.items():
        if (tb_name, col_name) in agg_joinable_columns:
            continue
        aggregated_cols = set()
        aggregated_cols.update(joinable_cols)
        for tb2, col2 in joinable_cols:
            aggregated_cols.update(joinable_columns[(tb2, col2)])
        for tb, col in aggregated_cols:
            other_tb_cols = []
            for tb2, col2 in aggregated_cols:
                if tb2 != tb:
                    other_tb_cols.append((tb2, col2))
            agg_joinable_columns[(tb, col)] = other_tb_cols
    return agg_joinable_columns


def find_joinable_columns(tb1: Table, tb2: Table) -> List[Tuple[str, str]]:
    max_coverage = 0
    cols = None
    for col1_name, col1 in tb1.columns.items():
        if (
            col1.stats["ratio of null values(%)"] > 20
            or col1.stats["number of distinct values"] < 5
        ):
            continue
        for col2_name, col2 in tb2.columns.items():
            if (
                col2.stats["ratio of null values(%)"] > 20
                or col2.stats["number of distinct values"] < 5
            ):
                continue
            if col1.name in tb1.primary_keys and col2.name in tb2.primary_keys:
                continue
            # calculate the coverage of the two columns
            sql1 = f"SELECT DISTINCT `{col1_name}` FROM `{tb1.tb_name}`;"
            vals1 = execute_sql_wrapper(sql1, tb1.db_path, 10)
            vals1 = [row[0] for row in vals1]
            sql2 = f"SELECT DISTINCT `{col2_name}` FROM `{tb2.tb_name}`;"
            vals2 = execute_sql_wrapper(sql2, tb2.db_path, 10)
            vals2 = [row[0] for row in vals2]
            coverage = len(set(vals1) & set(vals2)) / min(
                len(set(vals1)), len(set(vals2))
            )
            if coverage > 0.95 and coverage > max_coverage and is_same(col1, col2):
                max_coverage = coverage
                cols = (col1_name, col2_name)
    # print(
    #     f"[DEBUG] max_coverage: {max_coverage:.4f}, tb1: {tb1.tb_name}, tb2: {tb2.tb_name}, cols: {cols}"
    # )
    if max_coverage > 0.95:
        return cols
    else:
        return None


def is_same(col1: Column, col2: Column) -> bool:
    # let LLM determine whether the two columns refer to the same concept
    prompt = f"""Your task is to determine whether the two tables could be joined by the two columns.
Only if the two columns refer to the same concept, the two tables could be joined.
Please refer to both the column names along with the descriptions to determine whether the two tables could be joined by the two columns.
**column1:**
Table: {col1.tb_name}
Name: {col1.name}
Description: {col1.desp}
**column2:**
Table: {col2.tb_name}
Name: {col2.name}
Description: {col2.desp}
{col2.name}
Please answer "Yes" or "No" without any other explanation.
"""
    res = llm_check([prompt])[0]
    # print(f"[DEBUG] {prompt}\n res: {res.strip()}")
    return type(res) == str and res.startswith("Yes")


class Database:
    def __init__(self, dataset_name: str, db_name: str):
        self.db_path = self.get_db_path(dataset_name, db_name)
        self.db_name = db_name
        self.tables = self.load_tables(dataset_name, db_name)
        self.joinable_columns = self.aggregate_joinable_columns()

    def aggregate_joinable_columns(
        self,
    ) -> Dict[Tuple[str, str], List[Tuple[str, str]]]:
        joinable_columns = defaultdict(set)
        for tb_name, tb in self.tables.items():
            tb_joinable_columns = tb.joinable_columns
            for (tb_name, col_name), joinable_cols in tb_joinable_columns.items():
                for tb2, col2 in joinable_cols:
                    joinable_columns[(tb_name, col_name)].add((tb2, col2))
                    joinable_columns[(tb2, col2)].add((tb_name, col_name))
        agg_joinable_columns = update_joinable_columns(joinable_columns)
        # discover the joinable columns between two tables
        agg_joinable_columns = self.discover_more_joinable_columns(agg_joinable_columns)
        agg_joinable_columns = update_joinable_columns(agg_joinable_columns)
        return agg_joinable_columns

    def discover_more_joinable_columns(
        self, joinable_columns: Dict[Tuple[str, str], List[Tuple[str, str]]]
    ) -> Dict[Tuple[str, str], List[Tuple[str, str]]]:
        joinable_tables = defaultdict(set)
        for (tb_name, col_name), joinable_cols in joinable_columns.items():
            for tb2, col2 in joinable_cols:
                joinable_tables[tb_name].add(tb2)
        # check if the nonjoinable tables are joinable
        for tb_name in self.tables:
            non_joinable_tbs = set(self.tables.keys()) - joinable_tables.get(
                tb_name, set()
            )
            for tb2 in non_joinable_tbs:
                if tb2 == tb_name:
                    continue
                # calculate the coverage of the columns and select the best one
                cols = find_joinable_columns(self.tables[tb_name], self.tables[tb2])
                if cols is not None:
                    if (tb_name, cols[0]) in joinable_columns:
                        joinable_columns[(tb_name, cols[0])].append((tb2, cols[1]))
                    else:
                        joinable_columns[(tb_name, cols[0])] = [(tb2, cols[1])]
                    if (tb2, cols[1]) in joinable_columns:
                        joinable_columns[(tb2, cols[1])].append((tb_name, cols[0]))
                    else:
                        joinable_columns[(tb2, cols[1])] = [(tb_name, cols[0])]
        return joinable_columns

    def find_joinable_columns(self, tb1: Table, tb2: Table) -> List[Tuple[str, str]]:
        cols = []
        for col_name in tb1.columns:
            if col_name in tb2.columns:
                cols.append((col_name, col_name))
        return cols

    def get_db_path(self, dataset_name: str, name: str) -> str:
        if dataset_name == "birddev":
            db_dir = "../datasets/bird/dev/dev_databases"
        elif dataset_name == "spider":
            db_dir = "../datasets/spider/test_database"
        else:
            raise ValueError(f"Invalid dataset name: {dataset_name}")
        return os.path.join(db_dir, name, name + ".sqlite")

    def load_tables(self, dataset_name: str, db_name: str) -> Dict[str, Table]:
        tables = {}
        if dataset_name == "birddev":
            tb_desp_dir = (
                f"../datasets/bird/dev/dev_databases/{db_name}/database_description"
            )
            tb_desp_files = os.listdir(tb_desp_dir)
            for tb_desp_file in tb_desp_files:
                if not tb_desp_file.endswith(".csv"):
                    continue
                tb_desp = pd.read_csv(os.path.join(tb_desp_dir, tb_desp_file))
                tb_name = tb_desp_file.split(".")[0]
                tb_name = tb_name.lower()
                tb = Table(self.db_path, db_name, tb_name, tb_desp)
                tables[tb_name] = tb
        elif dataset_name == "spider":
            # read tables from sqlite file
            sql = f"SELECT name FROM sqlite_master WHERE type='table';"
            res = execute_sql_wrapper(sql, self.db_path, 10)
            for tb_name in res:
                tb = Table(self.db_path, db_name, tb_name[0].lower())
                tables[tb_name[0].lower()] = tb
        else:
            raise ValueError(f"Invalid dataset name: {dataset_name}")
        return tables

    def add_view(self, view_name: str):
        tb = Table(self.db_path, self.db_name, view_name)
        self.tables[view_name] = tb

    def remove_view(self, view_name: str):
        if view_name in self.tables:
            del self.tables[view_name]
        else:
            print(f"[Warning] View {view_name} not found in the database")

    def display(self):
        output = f"Database: {self.db_name}\n"
        for tb_name in self.tables:
            output += self.tables[tb_name].display()
        return output
