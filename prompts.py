from parser import SQLNode
from typing import List, Dict

rule_categories = [
    "Aggregation",
    "Sorting and Limiting",
    "Filtering",
    "Joining",
    "Data Integrity and Edge Cases",
    "Other",
    "Projection",
]


def intra_group_selection_prompt(
    question: str, evidence: str, schema_note: str, sql_nodes: List[SQLNode]
) -> str:
    prompt = f"""You are an expert in SQL and natural language understanding. Given a natural language query and multiple syntactically valid SQL queries that all produce the same execution result, your task is to select the best SQL query based on clarity, correctness, and alignment with the intent of the NL query.

Evaluation Criteria:
1. Readability & Maintainability: Favor simpler, more direct, and standard SQL constructs over unnecessarily complex or obscure ones.
2. Schema Usage: Use appropriate tables and columns as defined in the schema. Avoid overly complex or indirect formulations when a simpler one suffices.
3. Robustness: Ensure the query handles edge cases, tied results, and potential NULL values gracefully and consistently.

Input:
NL Query: {question}
Evidence: {evidence}
Schema: {schema_note}
Candidate SQL Queries:
"""
    for i, sql_node in enumerate(sql_nodes):
        prompt += f"SQL{i+1}: {sql_node.org_sql}\n"
    prompt += """Output Format (strictly follow this JSON format):
{{
"reason": "Briefly explain your reasoning process on selecting the best SQL in at most two sentences.",
"best_sql": "SQLk" // where k is the index of the best SQL (e.g., SQL1, SQL2, ..., SQLn)
}}
"""
    return prompt


def get_rule_generation_prompt(
    question: str,
    evidence: str,
    schema_note: str,
    sql1: str,
    sql2: str,
    sql_exec_notes: Dict[str, str],
    notes: str,
    correctness: int,
    historical_rules: List[str],
) -> str:
    base_info = f"""**Schema**
{schema_note}
    
**NL Query**
{question}
"""
    if evidence is not None and len(evidence.strip()) > 0:
        base_info += f"**Evidence**\n{evidence}\n"
    prompt = f"""You are an expert in SQL semantics and query intent analysis. Your task is to generate a single high-level, extremely concise, and informative hint that helps distinguish which of two SQL queries better matches the user’s natural language intent.

This hint will be used to guide future comparisons between SQL candidates. It must satisfy the following criteria:
1. Abstract & Generalizable:
- Do not reference specific table names, column names, values, or schema details.
- Capture a semantic principle applicable across diverse queries (e.g., about JOIN direction, aggregation scope, NULL handling, or filtering logic).
- Avoid overfitting to the specific case.
2. Rooted in Core Semantic Differences that lead to different execution results:
- Focus only on the essential discrepancy that explains why SQL1 is correct and SQL2 is incorrect.
- Avoid minor differences that do not fundamentally alter the query's correctness.
3. Output Format:
- Output exactly one sentence — no preamble, explanation, or elaboration.
- Use direct, imperative language.
- Strip every unnecessary word: make each word count.

{base_info}

**SQL1 (Correct)**
SQL: {sql1}
Notes on Execution Results: {sql_exec_notes[sql1]}

**SQL2 (Incorrect)**
SQL: {sql2}
Notes on Execution Results: {sql_exec_notes[sql2]}

**Previous LLM Comparison Notes**
{notes}
These notes led to an incorrect judgment, likely because they missed the core semantic issue.
"""
    if len(historical_rules) > 0:
        prompt += f"**Historical Hints(Insufficient)**\n{historical_rules}\nExisting hints failed to resolve this case; your hint must address the gap."

    prompt += "\nNow, analyze the fundamental reason SQL1 is correct while SQL2 is incorrect. Derive a general hint that would have prevented the earlier misjudgment. Output only the hint—no explanation, no preamble. Use clear, imperative language in a single sentence."
    return prompt


def get_rule_integration_prompt(rules: List[str]) -> str:
    # rules = "\n\n".join(f"Rule {i+1}: {rule}" for i, rule in enumerate(rules))
    prompt = f"""You are an expert in SQL semantics. Given a list of high-level hints for evaluating which of two SQL queries better aligns with a user’s intent, consolidate them into the most concise yet informative set of evaluation hints possible. Merge duplicates, remove redundancies, and distill the essential criteria into their simplest form—without introducing any new hints.
Hints:
{rules}

Output only the integrated hints, separated by " | ", with no additional text or explanation. Prioritize brevity while preserving all core criteria.
"""
    return prompt


def get_rule_integration_prompt2(rules: List[str]) -> str:
    rules = "\n".join(f"Hint {i+1}: {rule}" for i, rule in enumerate(rules))
    prompt = f"""You are an expert in SQL semantics. Given a list of hints related to evaluating which of two SQL queries aligns better with a user’s intent, your task is to remove duplicates without introducing any new information, or combining distinct content.

Your output must be a JSON dictionary organized as follows:
1. Each key is a comma-separated string of the original hint indices that were duplicates (e.g., "1,3").
2. Each value is the resulting hint, written clearly and without ambiguity.

Hints:
{rules}

Output format:
{{
    "hint_indices": "integrated_hint"
}}
"""
    return prompt


def get_rule_evaluation_prompt(
    question: str,
    evidence: str,
    schema_note: str,
    sql_node: SQLNode,
    rule: str,
) -> str:
    base_info = f"""**Schema**
{schema_note}
    
**NL Query**
{question}
"""
    if evidence is not None and len(evidence.strip()) > 0:
        base_info += f"**Evidence**\n{evidence}\n"
    prompt = f"""YYou are an expert in SQL and natural language understanding. Your task is to evaluate a given SQL statement against a specific constraint (Hint) and determine if the SQL violates that constraint.

### Instructions:
Please follow these steps to make your decision:
1. Context Analysis: Review the natural language query and database schema in the base information to understand the intended semantics of the query.
2. SQL Parsing: Analyze the structural components of the provided SQL statement (e.g., JOINs, aggregations, subqueries, conditions).
3. Hint Interpretation: Carefully read and understand the Hint.
4. Violation Check: Check if the SQL statement explicitly conflicts with or fails to satisfy the requirement defined in the Hint.    

{base_info}

**SQL**
{sql_node.org_sql}

**Hint**
{rule}

Output Format:
{{
    "reason": "Briefly explain how the SQL components either align with or conflict with the Hint. Keep it concise (under 50 tokens).",
    "violation": "Yes/No/Unsure"
}}
"""
    return prompt


def get_simple_comparison_prompt(
    base_info: str,
) -> str:
    prompt = f"""You are an expert in SQL and natural language understanding. Given a natural language query and two SQL statements that produce different results when executed, your task is to analyze and determine which SQL better aligns with the intent of the NL Query by focusing on their core differences.

{base_info}

Output Format:
{{
    "reason": "Briefly explain your reasoning process (under 150 tokens).",
    "better_sql": "SQL1/SQL2/Unsure",
}}
"""
    return prompt


def get_comparison_prompt(
    base_info: str,
    rules: List[str],
) -> str:
    rules = "\n".join(f"Hint {i+1}: {rule}" for i, rule in enumerate(rules))
    prompt = f"""You are an expert in SQL semantics and query intent analysis. Given a natural language (NL) query, database schema, optional evidence, and two SQL statements that yield different execution results, your task is to analyze and determine which SQL better aligns with the intent of the NL Query by focusing on their core logical differences. Ensure your explanation is logically consistent. Reference hints only when they meaningfully inform the decision (e.g., 'Hint 3 applies because...').

**Hints for Evaluation:**
The following hints are provided as reference heuristics to support your analysis. They reflect common pitfalls and best practices but are not absolute constraints. 
{rules}

{base_info}

Output strictly in the following JSON format with no extra text:
{{
    "reason": "Briefly explain your reasoning process (under 150 tokens).",
    "better_sql": "SQL1/SQL2/Unsure"
}}
"""
    return prompt


def get_generative_verifier_prompt(
    question: str,
    evidence: str,
    schema_note: str,
    sql_node: SQLNode,
) -> str:
    base_info = f"""**Schema**
{schema_note}
    
**NL Query**
{question}
"""
    prompt = f"""You are an expert in SQL and natural language understanding. Given a natural language query, database schema, optional evidence, and a SQL statement, your task is to determine whether the SQL statement is correct.

{base_info}
"""
    if evidence is not None and len(evidence.strip()) > 0:
        base_info += f"**Evidence**\n{evidence}\n"
    prompt += f"""**SQL**
{sql_node.org_sql}

Output Format:
{{
    "reason": "Briefly explain your reasoning process.",
    "correctness": "Yes/No/Unsure"
}}
"""
    return prompt


def get_rule_generality_and_clarity_evaluation_prompt(rule: str) -> str:
    prompt = f"""Evaluate the following SQL hint for its generality and clarity.
**Instructions:**
1.  **Generality:** Assess if the hint is broadly applicable to SQL correctness or too specific.
2.  **Clarity:** Assess if the hint is straightforward, easy to comprehend without requiring deep domain expertise, and free from ambiguous language or jargon.

Your output must be a JSON dictionary with the following structure:
{{
  "Generality_Reasoning": "Brief explanation of why the Generality is High or Low.",
  "Clarity_Reasoning": "Brief explanation of why the Clarity is High or Low."  
  "Generality": "High" | "Low",
  "Clarity": "High" | "Low",
}}

**SQL Hint to Evaluate:**
{rule}
"""
    return prompt


def get_rule_relevance_prompt(
    question: str, cond: str, hint: str, sqls: List[str]
) -> str:
    sqls = "\n".join(f"SQL {i+1}: {sql}" for i, sql in enumerate(sqls))
    prompt = f"""You are an expert SQL assistant. Your task is to determine whether a specific condition (a hint trigger) applies to a given NL query. You will make this determination based on the user's original natural language intent and the resulting SQL statements.

**NL Query:**
{question}

**SQL Set:**
{sqls}

**Hint Condition:**
{cond} 

**Hint Content:**
{hint}

Analysis Protocol:
1. Direct Mapping: Does the SQL explicitly implement a logic that matches the hint's trigger conditon?
2. Logical Necessity: Would the SQL be incorrect if this hint were ignored?

Instructions:
Evaluate if the Condition is a critical requirement for the given NL Query and SQL Set.
1. Answer "Yes" only if the condition is explicitly triggered by the query logic AND the SQL implementation.
2. Answer "No" if the condition is merely tangentially related, redundant, or if the SQL does not strictly require this specific hint to be valid.
3. If the relationship is ambiguous or the SQL only partially meets the criteria, answer "Unsure".

Output Format:
{{
    "reason": "Briefly explain your reasoning process (under 25 tokens).",
    "relevance": "Yes/No/Unsure"
}}
"""
    return prompt


def get_rule_condition_prompt(
    rule: str, query: str, evidence: str, sqls: List[str]
) -> str:
    prompt = f"""You are an expert SQL semantic analyst. Your task is to analyze a given SQL hint and generate a trigger condition in a JSON format. This trigger condition will consist of two parts: a natural language description and a list of relevant SQL keywords.

### Input Hint:
{rule}

For reference, here is the information of the case where the hint is obtained:
NL query: {query}
Evidenc: {evidence}
SQLs: {sqls}

### Instructions:
1. **NL Condition**: Create a concise and clear natural language phrase (maximum 20 words) that accurately describes the scenario where this hint is applicable.
2. **SQL Keywords**: List the minimal set of SQL keywords, clauses (e.g., NULL, COUNT, DISTINCT, LIMIT, RANK, ORDER BY, GROUP BY, HAVING, etc.) that, when present alone or together, strongly suggest the hint applies. 

### Output Format:
Return ONLY a valid JSON dictionary with the following structure. Do not include any explanation or markdown formatting.

{{
  "nl_condition": "<short, clear condition, ≤20 words>",
  "sql_keywords": ["keyword1", ..., "keywordn"]
}}
"""
    return prompt
