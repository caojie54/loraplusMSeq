#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

# Train, generate predictions from merged/, then score all three benchmarks.
# Infrastructure paths and GPU selection may be overridden by the caller.
BASE_MODEL_PATH=${BASE_MODEL_PATH:-meta-llama/Llama-3.1-8B} \
DATASET_PATH=${DATASET_PATH:-/data/user/cj/projects/CoMoL/datasets/natural_reasoning_20k} \
EVAL_DATA_PATH=${EVAL_DATA_PATH:-/data/user/cj/projects/CoMoL/datasets/natural_reasoning_eval} \
METHOD=static_random \
RANK=64 \
COMP_RATIO=0.02 \
SELECTION_INTERVAL=50 \
NUM_TRAIN_EPOCHS=1 \
SEED=1 \
LORA_OPTIMIZER_DTYPE=bf16 \
MODULE_OPTIMIZER_DTYPE=fp32 \
ALPHA_CANDIDATE_RATIO=0 \
LORA_LR=1e-4 \
MODULE_LR=1e-5 \
LORA_DROPOUT=0.0 \
MAX_LENGTH=1536 \
TRAIN_BATCH_SIZE=4 \
GRADIENT_ACCUMULATION_STEPS=8 \
WARMUP_STEPS=50 \
GRADIENT_CHECKPOINTING=false \
REQUIRED_FREE_GB=${REQUIRED_FREE_GB:-75} \
SAVE_MERGED_MODEL=true \
EVAL_AFTER_TRAIN=1 \
TEST_MAX_NEW_TOKENS=1536 \
TEST_BATCH_SIZE=64 \
NATURAL_REASONING_BENCHMARKS="gpqa_diamond math_500 mmlu_pro_500" \
bash task_natural_reasoning.sh
