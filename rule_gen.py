"""
Ask the LLM to generate rules for the comparison between the GT SQL and the generated SQLs
Only generate rules for the incorrect cases
"""

from parser import SQLCollection
from prompts import (
    get_rule_generation_prompt,
    get_rule_integration_prompt,
    get_rule_generality_and_clarity_evaluation_prompt,
)
from llm_infer import llm_check
from typing import Dict, List
from utils import parse_json


def rule_gen(
    qid_compare_results: Dict[int, Dict], qid_sql_collection: Dict[int, SQLCollection]
):
    for qid, compare_result in qid_compare_results.items():
        if "comparison_notes" not in compare_result:
            continue
        schema_note = compare_result["schema_note"]
        question = compare_result["question"]
        evidence = compare_result["evidence"]
        if len(qid_sql_collection[qid].rules) > 0:
            historical_rules = qid_sql_collection[qid].rules[-1]
        else:
            historical_rules = []
        sql_exec_notes = {}
        for sql_node in qid_sql_collection[qid].sql_nodes:
            sql_exec_notes[sql_node.org_sql] = sql_node.notes["exec_note"]
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
        if len(prompts) == 0:
            continue
        responses = llm_check(prompts, llm="deepseek")
        raw_rules = []
        for prompt, response in zip(prompts, responses):
            rule = response.strip()
            raw_rules.append(rule)
            print(f"[DEBUG]:\n[PROMPT]\n{prompt}\n[RULE]\n{rule}")
            print("-" * 100)
        # rule integration
        if len(raw_rules) > 1:
            prompt = get_rule_integration_prompt(raw_rules)
            response = llm_check([prompt], llm="deepseek")[0]
            rules = [rule.strip() for rule in response.strip().split("|")]
            print(f"[DEBUG] integrated rules:\n{rules}")
        else:
            rules = raw_rules
        qid_sql_collection[qid].rules.append(rules)
    return qid_sql_collection


def eval_rule_generality_and_clarity(rules: List[str]):
    rule_generality, rule_clarity = {}, {}
    prompts = []
    for rule in rules:
        prompt = get_rule_generality_and_clarity_evaluation_prompt(rule)
        prompts.append(prompt)
    responses = llm_check(prompts, llm="deepseek")
    for rule, response in zip(rules, responses):
        response = parse_json(response)
        if type(response) == dict:
            generality = response.get("Generality", "Low")
            clarity = response.get("Clarity", "Low")
        else:
            generality = "Low"
            clarity = "Low"
        rule_generality[rule] = 1 if generality == "High" else 0
        rule_clarity[rule] = 1 if clarity == "High" else 0
    return rule_generality, rule_clarity
