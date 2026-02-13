import os
import sys

method_name = sys.argv[1]
model_name = sys.argv[2]

rule_mode_list = ["none", "random"]
for rule_mode in rule_mode_list:
    selector = f"rule_{rule_mode}"
    cmd = f"python -u main.py --method_name {method_name} --model_name {model_name} --selector {selector} --rule_mode {rule_mode}"
    cmd += f" > logs/rule_debug/{method_name}-{model_name}-{rule_mode}.log"
    os.system(cmd)
