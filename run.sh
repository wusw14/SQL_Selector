python -u main.py --method_name alphasql --model_name Qwen3-30B --selector our_0121  > logs/multi-aspect/alphasql-30B-0121.log
python -u main.py --method_name alphasql --model_name Qwen2.5-7B --selector our_0121  > logs/multi-aspect/alphasql-7B-0121.log
python -u main.py --method_name sql-r1 --model_name 7B --selector our_0121  > logs/multi-aspect/sql-r1-7B-0121.log
python -u main.py --method_name sql-r1 --model_name 14B --selector our_0121  > logs/multi-aspect/sql-r1-14B-0121.log
python -u main.py --method_name alphasql --model_name Qwen2.5-32B --selector our_0121  > logs/multi-aspect/alphasql-32B-0121.log

# python -u eval.py --model_name 7B --method_name sql-r1 --align_column --selector our_0121 --llm_check --save_eval

# python -u eval.py --model_name 14B --method_name sql-r1 --align_column --selector our_0121 --llm_check --save_eval

# python -u eval.py --model_name Qwen3-30B --method_name alphasql --align_column --selector our_0121 --llm_check --save_eval

# python -u eval.py --model_name Qwen2.5-32B --method_name alphasql --align_column --selector our_0121 --llm_check --save_eval

# python -u eval.py --model_name Qwen2.5-7B --method_name alphasql --align_column --selector our_0121 --llm_check --save_eval
