import os

dataset_name = "birddev"

selector_list = [
    # "majority",
    # "GenRM",
    # "GenRM-CoT",
    "chase",
    "chase_deepseek",
    "ab_deepseek",
    "ab_deepseek_v2",
    # "agentar_v1,agentar_v2,agentar_v3",
    # "DPC",
    # "rule_samedb5_v1,rule_samedb5_v2,rule_samedb5_v3",
    # "rule_diffdb5_v1,rule_diffdb5_v2,rule_diffdb5_v3",
    # "rule_diffdb_0509_v1,rule_diffdb_0509_v2,rule_diffdb_0509_v3",
    # "ab_curation_v1,ab_curation_v2,ab_curation_v3",
    # "rule_4b2_v1,rule_4b2_v2,rule_4b2_v3",
    # "rule_9b2_v1,rule_9b2_v2,rule_9b2_v3",
    # "rule_diffdb_0511_v1,rule_diffdb_0511_v2,rule_diffdb_0511_v3",
    # "rule_0528_v1,rule_0528_v2,rule_0528_v3",
    # "ab_routing_v1,ab_routing_v2,ab_routing_v3",
    # "ab_fast_v1,ab_fast_v2,ab_fast_v3",
    # "ab_routing_v1,ab_routing_v2,ab_routing_v3",
    "ab_routing2_v1,ab_routing2_v2,ab_routing2_v3",
    # "deep_0609_v1,deep_0609_v2,deep_0609_v3",
    # "deep_0610_v1,deep_0610_v2,deep_0610_v3",
    # "ab_fast2_v1,ab_fast2_v2,ab_fast2_v3",
    # "ab_retrieval_emb2_v1,ab_retrieval_emb2_v2,ab_retrieval_emb2_v3",
    # "ab_retrieval_llm2_v1,ab_retrieval_llm2_v2,ab_retrieval_llm2_v3",
    # "rule_crossdb_v1,rule_crossdb_v2,rule_crossdb_v3",
    # "rule_crossds_v1,rule_crossds_v2,rule_crossds_v3",
    # "ab_curation2_v1,ab_curation2_v2,ab_curation2_v3",
    # "ab_size2_20_v1,ab_size2_20_v2,ab_size2_20_v3",
    # "ab_size2_50_v1,ab_size2_50_v2,ab_size2_50_v3",
    # "ab_size2_100_v1,ab_size2_100_v2,ab_size2_100_v3",
    # "ab_size2_200_v1,ab_size2_200_v2,ab_size2_200_v3",
]
# for model_name in ["7B", "14B"]:
#     for selector in selector_list:
#         cmd = f"python -u eval.py --model_name {model_name} --method_name sql-r1 --align_column --selector {selector} --save_eval --dataset_name {dataset_name}"
#         os.system(cmd)
#     print("=" * 60)

for model_name in ["Qwen2.5-7B", "Qwen3-30B"][1:]:
    for selector in selector_list:
        cmd = f"python -u eval.py --model_name {model_name} --method_name alphasql --align_column --selector {selector} --save_eval --dataset_name {dataset_name}"
        os.system(cmd)
    print("=" * 60)
