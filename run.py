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
        f"python -u agentar.py --dataset_name spidertest --method_name sql-r1 --model_name 7B --selector agentar{i}"
    )
    os.system(
        f"python -u agentar.py --dataset_name spidertest --method_name sql-r1 --model_name 14B --selector agentar{i}"
    )
    os.system(
        f"python -u agentar.py --dataset_name spidertest --method_name alphasql --model_name Qwen2.5-7B --selector agentar{i}"
    )
    os.system(
        f"python -u agentar.py --dataset_name spidertest --method_name alphasql --model_name Qwen3-30B --selector agentar{i}"
    )
