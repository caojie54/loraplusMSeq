log_num=0
task_floder=Med_eval/Qwen3.5-4B
model_name=Qwen/Qwen3.5-4B
port=28${log_num}35

eval_file=./eval_data/medbullets_op4.jsonl
python ./src/evaluation/eval.py --model_name $model_name  --eval_file $eval_file --port $port --strict_prompt --batch_size 1000 --max_new_tokens 2000 --task_floder $task_floder

eval_file=./eval_data/medbullets_op5.jsonl
python ./src/evaluation/eval.py --model_name $model_name  --eval_file $eval_file --port $port --strict_prompt --batch_size 1000 --max_new_tokens 2000 --task_floder $task_floder

eval_file=./eval_data/HLE_biomed.jsonl
python ./src/evaluation/eval.py --model_name $model_name  --eval_file $eval_file --port $port --strict_prompt --batch_size 1000 --max_new_tokens 2000 --task_floder $task_floder

eval_file=./eval_data/medqa_test.jsonl
python ./src/evaluation/eval.py --model_name $model_name  --eval_file $eval_file --port $port --strict_prompt --batch_size 1000 --max_new_tokens 2000 --task_floder $task_floder

eval_file=./eval_data/MedXpertQA_test.jsonl
python ./src/evaluation/eval.py --model_name $model_name  --eval_file $eval_file --port $port --strict_prompt --batch_size 1000 --max_new_tokens 2000 --task_floder $task_floder