#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}

export CUDA_VISIBLE_DEVICES=1

BASE_MODEL_PATH=${BASE_MODEL_PATH:-google/gemma-4-12B-it}
EVAL_DATA_PATH=${EVAL_DATA_PATH:-/data/user/cj/projects/loraplusMSeq/datasets/natural_reasoning_eval}
OUTPUT_ROOT=${OUTPUT_ROOT:-/data/user/cj/data2/experiments/natural-reasoning-eval}

TEST_MAX_NEW_TOKENS=${TEST_MAX_NEW_TOKENS:-2000}
TEST_BATCH_SIZE=${TEST_BATCH_SIZE:-300}
NATURAL_REASONING_BENCHMARKS=${NATURAL_REASONING_BENCHMARKS:-gpqa_diamond math_500 mmlu_pro_500}

MODEL_NAME=$(basename "${BASE_MODEL_PATH}")
OUTPUT_DIR="${OUTPUT_ROOT}/${MODEL_NAME}_sglang_2000_dosample"
python ../test_natural_reasoning_sglang.py \
  --model_path="${BASE_MODEL_PATH}" \
  --data_path="${EVAL_DATA_PATH}" \
  --benchmarks ${NATURAL_REASONING_BENCHMARKS} \
  --max_new_tokens="${TEST_MAX_NEW_TOKENS}" \
  --batch_size="${TEST_BATCH_SIZE}" \
  --tp_size=1 \
  --dp_size=1 \
  --mem_fraction_static 0.8 \
  --output_dir="${OUTPUT_DIR}" \
  --do_sample true

python ../evaluate_natural_reasoning.py \
  --prediction_dir "${OUTPUT_DIR}/predictions" \
  --benchmarks ${NATURAL_REASONING_BENCHMARKS}
