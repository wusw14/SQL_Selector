import os
import sys
from dotenv import set_key
import pynvml
import time


def find_available_port():
    if_available = False
    port = None
    while not if_available:
        for i in range(7):
            # if i == 5:
            #     continue
            # check the utilization of device i for every 10s
            if_available = True
            for t in range(30):
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                if utilization.gpu > 10:
                    if_available = False
                    break
                else:
                    time.sleep(2)
            if if_available:
                port = f"112{i}"
                break
        if port is None:
            time.sleep(600)
    print(f"Find port!!!: {port}")
    return port


retrieval_mode = "crosscase"
dataset_list = ["birddev", "spidertest"][:1]
selector_list = ["rule_0527_v1", "rule_0527_v2", "rule_0527_v3"]

for dataset in dataset_list:
    for model_name in ["Qwen2.5-7B", "Qwen3-30B"]:
        for selector in selector_list:
            # if os.path.exists(f"logs/{selector}/alphasql-{dataset}-{model_name}.log"):
            #     continue
            port = find_available_port()
            set_key(".env", "BASE_URL", f"http://localhost:{port}/v1")
            os.makedirs(f"logs/{selector}", exist_ok=True)
            cmd = f"nohup python -u online.py --method_name alphasql --dataset_name {dataset} --model_name {model_name} --selector {selector} --retrieval_mode {retrieval_mode} >> logs/{selector}/alphasql-{dataset}-{model_name}.log 2>&1 &"
            os.system(cmd)
            print(cmd)

for dataset in dataset_list:
    for model_name in ["7B", "14B"]:
        for selector in selector_list:
            # if os.path.exists(f"logs/{selector}/sql-r1-{dataset}-{model_name}.log"):
            #     continue
            port = find_available_port()
            set_key(".env", "BASE_URL", f"http://localhost:{port}/v1")
            os.makedirs(f"logs/{selector}", exist_ok=True)
            cmd = f"nohup python -u online.py --method_name sql-r1 --dataset_name {dataset} --model_name {model_name} --selector {selector} --retrieval_mode {retrieval_mode} >> logs/{selector}/sql-r1-{dataset}-{model_name}.log 2>&1 &"
            os.system(cmd)
            print(cmd)
