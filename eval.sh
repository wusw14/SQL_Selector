selector="our_0120"

python -u eval.py --model_name 7B --method_name sql-r1 --align_column --selector $selector --save_eval

python -u eval.py --model_name 14B --method_name sql-r1 --align_column --selector $selector --save_eval

python -u eval.py --model_name Qwen2.5-7B --method_name alphasql --align_column --selector $selector --save_eval

python -u eval.py --model_name Qwen2.5-32B --method_name alphasql --align_column --selector $selector --save_eval

python -u eval.py --model_name Qwen3-30B --method_name alphasql --align_column --selector $selector --save_eval