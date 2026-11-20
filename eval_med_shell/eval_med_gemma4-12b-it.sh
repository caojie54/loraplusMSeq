#!/usr/bin/env bash
set -euo pipefail
MEDREASON_ROOT=${MEDREASON_ROOT:-/data/user/cj/projects/loraplusMSeq/MedReason}
cd "${MEDREASON_ROOT}"

log_num=0
task_floder=Med_eval/gemma-4-12B-it
model_name=google/gemma-4-12B-it
port=28${log_num}35

# no  default chat_template
eval_file=./eval_data/medbullets_op4.jsonl
python ./evaluation/eval.py --model_name $model_name  --eval_file $eval_file --port $port --strict_prompt --batch_size 400 --max_new_tokens 2000 --task_floder $task_floder --results_dir "${RESULTS_DIR:-/data/user/cj/data2/MedReason}" --no-use_chat_template

eval_file=./eval_data/medbullets_op5.jsonl
python ./evaluation/eval.py --model_name $model_name  --eval_file $eval_file --port $port --strict_prompt --batch_size 400 --max_new_tokens 2000 --task_floder $task_floder --results_dir "${RESULTS_DIR:-/data/user/cj/data2/MedReason}" --no-use_chat_template

eval_file=./eval_data/HLE_biomed.jsonl
python ./evaluation/eval.py --model_name $model_name  --eval_file $eval_file --port $port --strict_prompt --batch_size 400 --max_new_tokens 2000 --task_floder $task_floder --results_dir "${RESULTS_DIR:-/data/user/cj/data2/MedReason}" --no-use_chat_template

eval_file=./eval_data/medqa_test.jsonl
python ./evaluation/eval.py --model_name $model_name  --eval_file $eval_file --port $port --strict_prompt --batch_size 400 --max_new_tokens 2000 --task_floder $task_floder --results_dir "${RESULTS_DIR:-/data/user/cj/data2/MedReason}" --no-use_chat_template

eval_file=./eval_data/MedXpertQA_test.jsonl
python ./evaluation/eval.py --model_name $model_name  --eval_file $eval_file --port $port --strict_prompt --batch_size 400 --max_new_tokens 2000 --task_floder $task_floder --results_dir "${RESULTS_DIR:-/data/user/cj/data2/MedReason}" --no-use_chat_template
