import os
import sys

# method_name = sys.argv[1]
# model_name = sys.argv[2]

# rule_mode_list = ["none", "random"]
# for rule_mode in rule_mode_list:
#     selector = f"rule_{rule_mode}"
#     cmd = f"python -u main.py --method_name {method_name} --model_name {model_name} --selector {selector} --rule_mode {rule_mode}"
#     cmd += f" > logs/rule_debug/{method_name}-{model_name}-{rule_mode}.log"
#     os.system(cmd)

for i in range(1, 4):
    os.system(
        f"python -u agentar.py --dataset_name spidertest --method_name sql-r1 --model_name 7B --selector agentar_v{i}"
    )
    os.system(
        f"python -u agentar.py --dataset_name spidertest --method_name sql-r1 --model_name 14B --selector agentar_v{i}"
    )
    os.system(
        f"python -u agentar.py --dataset_name spidertest --method_name alphasql --model_name Qwen2.5-7B --selector agentar_v{i}"
    )
    os.system(
        f"python -u agentar.py --dataset_name spidertest --method_name alphasql --model_name Qwen3-30B --selector agentar_v{i}"
    )

# dataset_list = ["birddev", "spidertest"][1:]
# selector_list = ["agentar1", "agentar2", "agentar3"]
# for dataset in dataset_list:
#     for model_name in ["7B", "14B"]:
#         for selector in selector_list:
#             cmd = f"python -u agentar.py --method_name sql-r1 --dataset_name {dataset} --model_name {model_name} --selector {selector}"
#             os.system(cmd)
#             print(cmd)

# for dataset in dataset_list:
#     for model_name in ["Qwen2.5-7B", "Qwen3-30B"]:
#         for selector in selector_list:
#             cmd = f"python -u agentar.py --method_name alphasql --dataset_name {dataset} --model_name {model_name} --selector {selector}"
#             os.system(cmd)
#             print(cmd)
