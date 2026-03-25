"""
1. Compare the generated SQL with GT SQL when the execution results are different
2. Generate the comparison notes for the generated SQLs
"""

from typing import List, Dict
from prompts import get_simple_comparison_prompt, get_comparison_prompt
from llm_infer import llm_check
from utils import parse_json
from selection import SQLNode
from representation import get_relevant_rules
from parser import SQLCollection, SQLNode
from copy import deepcopy


def get_prompt(
    base_info: str,
    sql_node1: SQLNode,
    sql_node2: SQLNode,
    rule_mode: str,
    rules: List[str],
) -> str:
    base_info2 = f"""{base_info}

**SQL1**
{sql_node1.org_sql}
Notes on Execution Results: {sql_node1.notes["exec_note"]}
"""
    if (
        sql_node1.notes["join_note"] is not None
        and len(sql_node1.notes["join_note"]) > 0
    ):
        base_info2 += f"Notes on Join Conditions: {sql_node1.notes['join_note']}\n"
    base_info2 += f"""\n\n**SQL2**
{sql_node2.org_sql}
Notes on Execution Results: {sql_node2.notes["exec_note"]}
"""
    if (
        sql_node2.notes["join_note"] is not None
        and len(sql_node2.notes["join_note"]) > 0
    ):
        base_info2 += f"Notes on Join Conditions: {sql_node2.notes['join_note']}\n"
    if rule_mode == "none":
        comparison_prompt = get_simple_comparison_prompt(base_info2)
    else:
        relevant_rules = []
        for i, rule in enumerate(rules):
            if sql_node1.score_each_rule[i] == 0 or sql_node2.score_each_rule[i] == 0:
                relevant_rules.append(rule)
        if len(relevant_rules) > 0:
            comparison_prompt = get_comparison_prompt(base_info2, relevant_rules)
        else:
            comparison_prompt = get_simple_comparison_prompt(base_info2)
    return comparison_prompt


def gen_comparison_notes(
    question: str,
    evidence: str,
    schema_note: str,
    gt_sql_node: SQLNode,
    gen_sql_nodes: List[SQLNode],
    rule_mode: str,
    rules: List[str] = None,
):
    """
    Input: a list of preds need to be compared with the GT SQL
    Output: comparison notes
    """
    base_info = f"""**Schema**
{schema_note}

**NL Query**
{question}
"""
    if evidence is not None and len(evidence.strip()) > 0:
        base_info += f"**Evidence**\n{evidence}\n"
    prompts = []
    for gen_sql_node in gen_sql_nodes:
        prompt = get_prompt(base_info, gt_sql_node, gen_sql_node, rule_mode, rules)
        prompts.append(prompt)
    # print(f"[DEBUG] [Prompt Example]\n{prompts[0]}")
    responses = llm_check(prompts, llm="Qwen3-30B")
    comparison_notes = []
    for i, response in enumerate(responses):
        note = parse_json(response)
        try:
            correctness = int(note["better_sql"] == "SQL1")
        except:
            if "better_sql" in note and type(note) == str:
                index = note.rindex("better_sql")
                if "SQL1" in note[index:]:
                    correctness = 1
                else:
                    correctness = 0
            else:
                correctness = 0.5
        comparison_notes.append(
            {
                "SQL1": gt_sql_node.org_sql,
                "SQL2": gen_sql_nodes[i].org_sql,
                "comparison_note": note,
                "correctness": correctness,
            }
        )
    return comparison_notes


def binary_comparison(
    qids,
    qid_sql_collection: Dict[str, SQLCollection],
    rule_mode,
    representation_model,
    db_qids: Dict[str, List[int]],
):
    # qid_sql_nodes is the SQL nodes for the question after the intra group selection
    qid_rules = {}
    for qid, sql_collection in qid_sql_collection.items():
        try:
            qid_rules[qid] = sql_collection.rules[-1]
        except:
            pass
    # print(f"len(qid_rules): {len(qid_rules)}")
    # for qid, rules in qid_rules.items():
    #     print(f"qid: {qid}, rules: {rules}")
    results = {}
    for qid in qids:
        sql_collection = qid_sql_collection[qid]
        gt_nodes = sql_collection.gt_sql_nodes
        sql_nodes = deepcopy(sql_collection.incorrect_sql_nodes)
        if (
            gt_nodes is None
            or len(gt_nodes) == 0
            or sql_nodes is None
            or len(sql_nodes) == 0
        ):
            continue
        info = sql_collection.info
        question = info["question"]
        evidence = info["evidence"]
        schema_note = sql_collection.schema_note
        preds = sql_collection.sqls
        comparison_notes_all = []
        for gt_node in gt_nodes:
            if rule_mode != "none":
                rules = get_relevant_rules(
                    qid,
                    qid_rules,
                    preds,
                    representation_model,
                    rule_mode,
                    top_k=5,
                    exclude_qids=db_qids[sql_collection.db.db_name],
                )
            else:
                rules = None
            comparison_notes = gen_comparison_notes(
                question,
                evidence,
                schema_note,
                gt_node,
                sql_nodes,
                rule_mode,
                rules,
            )
            comparison_notes_all.extend(comparison_notes)
        result = dict(info)
        result["schema_note"] = schema_note
        result["comparison_notes"] = comparison_notes_all
        results[qid] = result
    return results
