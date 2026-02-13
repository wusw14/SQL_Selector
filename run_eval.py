import os

dataset_name = "birddev"

selector_list = [
    "majority",
    "minbug",
    "exec",
    "join",
    # "our_0120",
    # "our_0121",
    # "our_0126",
    # "our_0127",
]
for model_name in ["7B", "14B"]:
    for selector in selector_list:
        cmd = f"python -u eval.py --model_name {model_name} --method_name sql-r1 --align_column --selector {selector} --save_eval --dataset_name {dataset_name}"
        os.system(cmd)
    print("=" * 100)

for model_name in ["Qwen2.5-7B", "Qwen2.5-32B", "Qwen3-30B"]:
    for selector in selector_list:
        cmd = f"python -u eval.py --model_name {model_name} --method_name alphasql --align_column --selector {selector} --save_eval --dataset_name {dataset_name}"
        os.system(cmd)
    print("=" * 100)
