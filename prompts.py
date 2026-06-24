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
2. Logical Necessity: Would applying this hint make the resulting SQL more accurate and robust, prevent a common data logic error, or resolve an ambiguity? (Note: A hint acts as expert domain knowledge).

Instructions:
Evaluate if the Condition is a critical requirement for the given NL Query and SQL Set.
1. Answer "Yes" only if the condition is explicitly triggered by the query logic AND the SQL implementation.
2. Answer "No" if the condition is not related to the query logic or the SQL implementation.
3. If the relationship is ambiguous, answer "Unsure".

Output Format:
{{
    "reason": "Briefly explain your reasoning process (under 25 tokens).",
    "relevance": "Yes/No/Unsure"
}}
"""
    return prompt


def get_rule_rank_prompt(question, hints, sqls):
    k = len(sqls)
    hint_str = "\n".join(
        [
            f"Hint {i}: {hint.text}, Trigger Condition: {hint.nl_cond}"
            for i, hint in enumerate(hints)
        ]
    )
    prompt = f"""Your task is to identify the critical hints that help distinguish the correct SQL query from a set of candidates based on a natural language (NL) question.

##Instructions:## 
1. Analyze the provided NL Query and the Candidate SQL Set.
2. Review the list of Hints provided.
3. Select the hints that are effective at resolving the logic ambiguities between the candidate SQLs. and whose trigger conditions are relevant to the NL query.
4. Ignore hints that do not help in choosing between the specific candidate SQLs provided.
5. **CRITICAL SELECTION LIMIT**: You can select a MAXIMUM of 5 hints. If fewer than 5 hints are relevant, only select the relevant ones. Do not force selection to reach 5. If more than 5 hints seem applicable, prioritize and select up to 5 of the most impactful ones that resolve the core SQL ambiguities.

##Input Data:##
NL Query: {question}

Candidate SQLs: 
{sqls}

Hints:
{hint_str}

Output Requirement:
Return a JSON dictionary with exactly two keys:
- reason: A brief explanation of why these specific hints were chosen.
- hint_indices: A list of integers representing the indices of the selected hints.

#Output Format:#
{{
  "reason": "brief explaination of why these specific hints were chosen (under 150 tokens).",
  "hint_indices": [int, int, ...]
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


def get_rule_logic_cond(rule: str, query: str, evidence: str, sqls: List[str]) -> str:
    prompt = f"""You are an expert SQL analysis assistant. Your task is to extract the logical condition (pattern) from a given SQL Hint and its corresponding context (NL query, Evidence, and SQL examples) that triggers the Hint.

### TASK DESCRIPTION
Analyze when the given "Hint" should be considered based on the SQL patterns observed in the context. You need to identify what SQL keywords, functions, or operators strongly indicate that the Hint is applicable.

### FORMATTING RULES FOR `logic_cond`
1. Separate alternative independent conditions using the pipe symbol `|` (denoting OR).
2. Within a single condition, if MULTIPLE keywords MUST appear together to trigger the hint, connect them using the ampersand symbol `&` (denoting AND). 
3. IMPORTANT: Do NOT overuse `&`. If a single SQL keyword/function alone is sufficient to trigger the pattern, just list that keyword without `&`. 
4. Keywords should be uppercase (e.g., `ROUND`, `LIMIT`, `LIKE`).
5. The condition MUST ONLY consist of generic SQL keywords, operators, or functions. 
   - **NEVER** include specific table names (e.g., `Patient`, `Laboratory`).
   - **NEVER** include specific column names (e.g., `U-PRO`, `Age`, `Birthday`).
   - **NEVER** include specific literal values (e.g., `1984`, `'SLE'`, `50`).

#### For `nl_cond` (NL Scenario):
1. **ONE SENTENCE ONLY**: Summarize the rule or business scenario in a single, high-quality English sentence.
2. **DOMAIN AGNOSTIC**: Focus on the underlying computational logic (e.g., "calculating percentages", "ranking items", "finding historical ages") rather than domain-specific entities (e.g., don't mention "cards", "molecules", or "patients").

### EXPECTED OUTPUT FORMAT
You must output JSON ONLY. Do not wrap the JSON in any markdown code blocks (like ```json). The structure must be a JSON dictionary with the following keys:
{{
  "thought": "Step-by-step reasoning showing how you derived both conditions by stripping away domain-specific tables/columns/values.",
  "logic_cond": "The formatted abstract SQL logic condition string.",
  "nl_cond": "The one-sentence generic natural language condition string."
}}

### EXAMPLES

Example 1:
Context:
- Hint: Prefer handling ties explicitly to avoid arbitrarily discarding valid results.
- NL Query: What is the complete address of the school with the lowest excellence rate?
- SQLs: ['... ORDER BY ... LIMIT 1', '... ORDER BY ... ROW_NUMBER() ...']
Output:
{{
  "thought": "The user wants the 'lowest' item, which implies a top-1 ranking scenario where ties might occur. In SQL, this is implemented by sorting and truncating, or using row numbering without handling ties. I will exclude domain terms like 'school' or 'excellence rate'.",
  "logic_cond": "ORDER BY & LIMIT | ORDER BY & ROW_NUMBER",
  "nl_cond": "When the query asks for the extreme highest or lowest records (top-N) where identical values or ties might exist."
}}

Example 2:
Context:
- Hint: Prefer exact equality over pattern matching for categorical or specific values.
- NL query: Among the Artifact cards, which are black color?
- SQLs: ['... WHERE c.originaltype = \'Artifact\'', '... WHERE c.originaltype LIKE \'%Artifact%\'']
Output:
{{
  "thought": "The intent is to filter by definitive categories or codes. The SQL pattern switches between exact match (=) and fuzzy match (LIKE). I will keep the descriptions generic without mentioning 'cards' or 'color'.",
  "logic_cond": "= | LIKE",
  "nl_cond": "When filtering records based on strict, predefined categories, discrete codes, or exact match attributes."
}}

### YOUR TURN
Now, analyze the following case and output the JSON dictionary.

Context:
- Hint: {rule}
- NL query: {query}
- Evidence: {evidence}
- SQLs: {sqls}
Output:
"""
    return prompt
