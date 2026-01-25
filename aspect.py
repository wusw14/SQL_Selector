from utils import parse_json

OUTPUT_FORMAT = """{{
    "reason": "Provide a brief explanation of your reasoning process while comparing the {aspect_name} operations in the two SQL statements.",
    "conclusion": "Yes/No",
    "better_sql": "SQL1/SQL2/Both",
}}"""

GROUP_BY_TASK_DESP = """You are provided with a natural language query (NL Query) and two SQL statements that produce different results when executed. Additionally, hints are provided to help you compare the GROUP BY operations in the two SQL statements. Your task is to evaluate whether the GROUP BY operation is the fundamental reason for the difference in execution results between the two SQL statements.

Instructions:
1. Analyze and compare the GROUP BY operations in the two SQL statements.
2. Determine whether the GROUP BY operation is the primary reason for the difference in execution results between the two SQL statements.
3. Based on your analysis:
    - If the GROUP BY operation is the fundamental reason for the difference, conclude "Yes" and identify the better SQL statement.
    - Otherwise, conclude "No" and select "Both" as the better SQL.
"""

GROUP_BY_OUTPUT_FORMAT = OUTPUT_FORMAT.format(aspect_name="GROUP BY")

ORDER_BY_TASK_DESP = """You are provided with a natural language query (NL Query) and two SQL statements that produce different results when executed. The hints are important to help you do the analysis. Your task is to evaluate whether the ranking operation is the fundamental reason for the difference in execution results between the two SQL statements.

Instructions:
1. Focus specifically on the ranking operation and the object of the ranking operation (i.e., the ORDER BY clause, or any use of MAX/MIN) in both SQL statements.
2. Determine whether the ranking operation is the primary reason for the difference in execution results between the two SQL statements.
3. Based on your analysis:
    - If the ranking operation is the fundamental reason for the difference, conclude "Yes" and identify the better SQL statement.
    - Otherwise, conclude "No" and select "Both" as the better SQL.
"""

ORDER_BY_OUTPUT_FORMAT = OUTPUT_FORMAT.format(aspect_name="RANKING")

FILTERING_TASK_DESP = """You are provided with a natural language query (NL Query) and two SQL statements that produce different results when executed. Your task is to determine whether the filtering conditions are the fundamental reason for the difference in execution results between the two SQL statements. 

Instructions:
1. Compare the filtering conditions in the two SQL statements.
2. Determine whether the filtering conditions are the fundamental reason for the difference in execution results between the two SQL statements.
3. Based on your analysis:
- If yes, conclude "Yes" and identify the better SQL statement by evaluating:
    - The correctness of the filtering conditions (e.g., whether they align with the NL query or are consistent with the evidence).
    - The necessity of the filtering conditions in fulfilling the NL query.
- Otherwise, conclude "No" and select "Both" as the better SQL.
"""

FILTERING_OUTPUT_FORMAT = OUTPUT_FORMAT.format(aspect_name="FILTERING")

RETURNED_COLUMNS_TASK_DESP = """You are provided with a natural language query (NL Query) and two SQL statements that produce different results when executed. Additionally, descriptions of the returned columns in two SQLs are provided. Your task is to which SQL returns more appropriate columns to meet the intent of the NL Query.

Instructions:
1. Compare the returned columns of the two SQLs.
2. Determine which SQL returns more appropriate columns to meet the intent of the NL Query.
"""

RETURNED_COLUMNS_OUTPUT_FORMAT = """{{
    "reason": "Briefly explain your reasoning process on comparing the returned columns of the two SQLs.",
    "better_sql": "SQL1/SQL2/Both",
}}"""

TaskDesp = {
    "group_by": GROUP_BY_TASK_DESP,
    "order_by": ORDER_BY_TASK_DESP,
    "filtering": FILTERING_TASK_DESP,
    "returned_columns": RETURNED_COLUMNS_TASK_DESP,
}

OutputFormat = {
    "group_by": GROUP_BY_OUTPUT_FORMAT,
    "order_by": ORDER_BY_OUTPUT_FORMAT,
    "filtering": FILTERING_OUTPUT_FORMAT,
    "returned_columns": RETURNED_COLUMNS_OUTPUT_FORMAT,
}


class Aspect:
    def __init__(self, name: str, base_info: str, hint: str):
        self.name = name
        self.task_desp = TaskDesp[name]
        self.base_info = base_info
        self.hint = hint
        self.output_format = OutputFormat[name]

    def get_prompt(self) -> str:
        return f"""{self.task_desp}

{self.base_info}

**Hint**
{self.hint}

Output Format:
{self.output_format}
"""

    def parse_output(self, output: str, other_info=None) -> str:
        # based on the LLM output, determine which SQL is the better one
        res = parse_json(output)
        if type(res) == str:
            return "Both"
        if self.name == "returned_columns":
            return res["better_sql"]
        elif self.name in ["group_by", "order_by", "filtering"]:
            conclusion = res["conclusion"]
            if conclusion == "Yes":
                return res["better_sql"]
        # elif self.name == "filtering":
        #     scores = 0
        #     for filtering_condition, conclusion in res.items():
        #         if conclusion == "Yes":
        #             score = 1
        #         elif conclusion == "No":
        #             score = -1
        #         if filtering_condition in other_info["SQL1"]:
        #             scores += score
        #         elif filtering_condition in other_info["SQL2"]:
        #             scores -= score
        #     if scores > 0:
        #         return "SQL1"
        #     elif scores < 0:
        #         return "SQL2"
        #     else:
        #         return "Both"
        else:
            raise ValueError(f"Invalid aspect name: {self.name}")
