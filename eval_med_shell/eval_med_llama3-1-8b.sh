log_num=0
task_floder=Med_eval/Llama-3.1-8B
model_name=meta-llama/Llama-3.1-8B
port=28${log_num}35

# no  default chat_template
eval_file=./eval_data/medbullets_op4.jsonl
python ./src/evaluation/eval.py --model_name $model_name  --eval_file $eval_file --port $port --strict_prompt --batch_size 1000 --max_new_tokens 2000 --task_floder $task_floder --no-use_chat_template 

eval_file=./eval_data/medbullets_op5.jsonl
python ./src/evaluation/eval.py --model_name $model_name  --eval_file $eval_file --port $port --strict_prompt --batch_size 1000 --max_new_tokens 2000 --task_floder $task_floder --no-use_chat_template 

eval_file=./eval_data/HLE_biomed.jsonl
python ./src/evaluation/eval.py --model_name $model_name  --eval_file $eval_file --port $port --strict_prompt --batch_size 1000 --max_new_tokens 2000 --task_floder $task_floder --no-use_chat_template 

eval_file=./eval_data/medqa_test.jsonl
python ./src/evaluation/eval.py --model_name $model_name  --eval_file $eval_file --port $port --strict_prompt --batch_size 1000 --max_new_tokens 2000 --task_floder $task_floder --no-use_chat_template 

eval_file=./eval_data/MedXpertQA_test.jsonl
python ./src/evaluation/eval.py --model_name $model_name  --eval_file $eval_file --port $port --strict_prompt --batch_size 1000 --max_new_tokens 2000 --task_floder $task_floder --no-use_chat_template 