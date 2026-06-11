#!/usr/bin/env bash
set -euo pipefail

source /mnt/petrelfs/caojie1/anaconda3/etc/profile.d/conda.sh
conda activate comol

MODEL_DIR=/mnt/dhwfile/raise/user/caojie/loraplusMSeq/outputs/commonsense170k/llama-3-1-8b-lora-qkvogateupdown-rank32-commonsense170k-epoch2

python test_commonsense.py \
  --model_path "${MODEL_DIR}" \
  --data_path /mnt/petrelfs/caojie1/projects/CoMoL/datasets/math_commonsense \
  --max_new_tokens 64 \
  --batch_size 256

python evaluate_commonsense.py \
  --predict_file "${MODEL_DIR}/predictions/boolq_responses.jsonl"