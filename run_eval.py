import os

dataset_name = "spidertest"

selector_list = [
    "majority",
    "GenRM",
    "GenRM-CoT",
    "chase",
    "agentar_v1,agentar_v2,agentar_v3",
    "rule_samedb_v1,rule_samedb_v2,rule_samedb_v3",
    "rule_diffdb_v1,rule_diffdb_v2,rule_diffdb_v3",
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
