import os

dataset_name = "spidertest"

selector_list = [
    "majority",
    # "GenRM",
    # "GenRM-CoT",
    # "chase",
    # "agentar1",
    # "agentar2",
    # "agentar3",
    # "hint_random",
    # "hint_wo_keyword",
    # "hint_wo_semantic",
    # "rule_0420",
]
for model_name in ["7B", "14B"]:
    for selector in selector_list:
        cmd = f"python -u eval.py --model_name {model_name} --method_name sql-r1 --align_column --selector {selector} --save_eval --dataset_name {dataset_name}"
        os.system(cmd)
    print("=" * 100)

for model_name in ["Qwen2.5-7B", "Qwen3-30B"]:
    for selector in selector_list:
        cmd = f"python -u eval.py --model_name {model_name} --method_name alphasql --align_column --selector {selector} --save_eval --dataset_name {dataset_name}"
        os.system(cmd)
    print("=" * 100)
