"""
Ask the LLM to generate rules for the comparison between the GT SQL and the generated SQLs
Only generate rules for the incorrect cases
"""

from parser import SQLCollection, SQLNode
from prompts import get_rule_generation_prompt, get_rule_integration_prompt
from llm_infer import llm_check
from typing import Dict


def rule_generation(
    qid_compare_results: Dict[str, Dict], qid_sql_collection: Dict[str, SQLCollection]
):
    for qid, compare_result in qid_compare_results.items():
        if "comparison_notes" not in compare_result:
            continue
        schema_note = compare_result["schema_note"]
        question = compare_result["question"]
        evidence = compare_result["evidence"]
        sql_exec_notes = compare_result["sql_exec_notes"]
        historical_rules = qid_sql_collection[qid].rules
        # only generate rules for the incorrect cases
        comparison_notes = compare_result["comparison_notes"]
        # generate the rules for each incorrect comparison
        prompts = []
        for comparison_note in comparison_notes:
            if comparison_note["correctness"] == 1:
                continue
            sql1 = comparison_note["SQL1"]
            sql2 = comparison_note["SQL2"]
            notes = comparison_note["comparison_note"]
            correctness = comparison_note["correctness"]
            prompt = get_rule_generation_prompt(
                question,
                evidence,
                schema_note,
                sql1,
                sql2,
                sql_exec_notes,
                notes,
                correctness,
                historical_rules,
            )
            prompts.append(prompt)
        responses = llm_check(prompts, llm="deepseek")
        raw_rules = []
        for prompt, response in zip(prompts, responses):
            rule = response.strip()
            raw_rules.append(rule)
        # rule integration
        prompt = get_rule_integration_prompt(raw_rules)
        response = llm_check(prompt, llm="deepseek")
        rule = response.strip()
        qid_sql_collection[qid].rules.append(rule)
    return qid_sql_collection
