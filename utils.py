import sqlite3
import sys
from func_timeout import func_timeout, FunctionTimedOut
import json


def execute_sql_wrapper(sql, db_path, timeout, return_columns: bool = False):
    try:
        res = func_timeout(timeout, execute_sql, args=(sql, db_path, return_columns))
    except FunctionTimedOut:
        print(f"SQL:\n{sql}\nTime Out!")
        print("-" * 30)
        res = "Time Out"
        if return_columns:
            res = (res, [])
    except Exception as e:
        res = "Unexecutable"
        if return_columns:
            res = (res, [])
    return res


def execute_sql(sql: str, db_path: str, return_columns: bool = False):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute(sql)
        result = cursor.fetchall()
        # filter row if all the values in this row are None
        result = [
            row
            for row in result
            if not all(value is None or value == "" for value in row)
        ]
        if return_columns:
            columns = [description[0] for description in cursor.description]
            result = (result, columns)
    except:
        if return_columns:
            result = ("Unexecutable", [])
        else:
            result = "Unexecutable"
    cursor.close()
    conn.close()
    return result


def parse_json(result: str) -> int:
    start_idx = result.find("{")
    end_idx = result.rfind("}") + 1
    result = result[start_idx:end_idx]
    try:
        result = json.loads(result)
        return result
    except:
        return result


def parse_result(result: str) -> str:
    start_idx = result.find("{")
    end_idx = result.rfind("}") + 1
    result = result[start_idx:end_idx]
    try:
        result = json.loads(result)
        return result["better_sql"]
    except:
        return "Unsure"
