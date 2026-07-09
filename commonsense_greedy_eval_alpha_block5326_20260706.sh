#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

source /mnt/petrelfs/caojie1/anaconda3/etc/profile.d/conda.sh
conda activate comol

export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export HF_HOME=${HF_HOME:-/mnt/dhwfile/raise/user/caojie/huggingface}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}

MODEL_DIR=/mnt/dhwfile/raise/user/caojie/loraplusMSeq/outputs/commonsense170k/llama-3-1-8b-seq-alpha-qkvogateupdown-rank64-commonsense170k-epoch1-ratio0.02-block5326-loralr1e-4-modulelr1e-5
DATA_PATH=/mnt/petrelfs/caojie1/projects/CoMoL/datasets/math_commonsense

python test_commonsense.py \
  --model_path "${MODEL_DIR}" \
  --data_path "${DATA_PATH}" \
  --max_new_tokens 64 \
  --batch_size 256 \
  --do_sample false

python evaluate_commonsense.py \
  --predict_file "${MODEL_DIR}/predictions/boolq_responses.jsonl"
