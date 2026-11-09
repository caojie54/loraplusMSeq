#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

source /mnt/petrelfs/caojie1/anaconda3/etc/profile.d/conda.sh
conda activate loraplusm

export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export HF_HOME=${HF_HOME:-/mnt/dhwfile/raise/user/caojie/huggingface}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}

BASE_MODEL_PATH=${BASE_MODEL_PATH:-google/gemma-4-E2B-it}
EVAL_DATA_PATH=${EVAL_DATA_PATH:-/mnt/petrelfs/caojie1/projects/CoMoL/datasets/natural_reasoning_eval}
OUTPUT_ROOT=${OUTPUT_ROOT:-/mnt/dhwfile/raise/user/caojie/loraplusMSeq/outputs/natural-reasoning-eval}

TEST_MAX_NEW_TOKENS=${TEST_MAX_NEW_TOKENS:-1536}
TEST_BATCH_SIZE=${TEST_BATCH_SIZE:-144}
NATURAL_REASONING_BENCHMARKS=${NATURAL_REASONING_BENCHMARKS:-gpqa_diamond math_500 mmlu_pro_500}

OUTPUT_DIR="${OUTPUT_ROOT}/gemma-4-E2B-it"
python test_natural_reasoning.py \
  --model_path="${BASE_MODEL_PATH}" \
  --data_path="${EVAL_DATA_PATH}" \
  --benchmarks ${NATURAL_REASONING_BENCHMARKS} \
  --max_new_tokens="${TEST_MAX_NEW_TOKENS}" \
  --batch_size="${TEST_BATCH_SIZE}" \
  --output_dir="${OUTPUT_DIR}"

python evaluate_natural_reasoning.py \
  --prediction_dir "${OUTPUT_DIR}/predictions" \
  --benchmarks ${NATURAL_REASONING_BENCHMARKS}
