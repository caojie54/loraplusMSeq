#!/usr/bin/env bash
set -euo pipefail

cd /mnt/petrelfs/caojie1/projects/loraplusMSeq

source /mnt/petrelfs/caojie1/anaconda3/etc/profile.d/conda.sh
conda activate loraplusm

export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export HF_HOME=${HF_HOME:-/mnt/dhwfile/raise/user/caojie/huggingface}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}

MODEL_DIR=/mnt/dhwfile/raise/user/caojie/loraplusMSeq/outputs/natural-reasoning-20k-gemma-4-E2B/gemma-4-E2B-seq-alpha-gradnorm-pressure-qkvogateupdown-rank64-natural-reasoning-20k-epoch1-ratio0.02-block50-seed2-loraoptbf16-moduleoptfp32-resetoffload-loralr1e-4-modulelr1e-5
EVAL_DATA_PATH=${EVAL_DATA_PATH:-/mnt/petrelfs/caojie1/projects/CoMoL/datasets/natural_reasoning_eval}
TEST_MAX_NEW_TOKENS=${TEST_MAX_NEW_TOKENS:-1536}
TEST_BATCH_SIZE=${TEST_BATCH_SIZE:-144}
NATURAL_REASONING_BENCHMARKS=${NATURAL_REASONING_BENCHMARKS:-gpqa_diamond math_500 mmlu_pro_500}

python test_natural_reasoning.py \
  --model_path="${MODEL_DIR}" \
  --data_path="${EVAL_DATA_PATH}" \
  --benchmarks ${NATURAL_REASONING_BENCHMARKS} \
  --max_new_tokens="${TEST_MAX_NEW_TOKENS}" \
  --batch_size="${TEST_BATCH_SIZE}"

python evaluate_natural_reasoning.py \
  --prediction_dir "${MODEL_DIR}/predictions" \
  --benchmarks ${NATURAL_REASONING_BENCHMARKS}
