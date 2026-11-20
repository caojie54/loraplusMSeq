#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

# Restrict candidates if needed: CUDA_VISIBLE_DEVICES=1,2 bash <this script>
export BASE_MODEL_PATH=${BASE_MODEL_PATH:-google/gemma-4-12B}
export DATASET_PATH=${DATASET_PATH:-/data/user/cj/projects/loraplusMSeq/datasets/natural_reasoning_20k}
export OUTPUT_ROOT=${OUTPUT_ROOT:-/data/user/cj/data2/experiments/loraplusMSeq/outputs/natural-reasoning-20k/gemma-4-12b}
export METHOD=lora
export RANK=${RANK:-64}
export COMP_RATIO=${COMP_RATIO:-0.02193}
export NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-2}
export SEED=${SEED:-0}
export LORA_OPTIMIZER_DTYPE=${LORA_OPTIMIZER_DTYPE:-bf16}
export LORA_LR=${LORA_LR:-1e-4}
export SAVE_MERGED_MODEL=true
export MAX_LENGTH=${MAX_LENGTH:-1400}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-2}
export GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-16}
export GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-false}
export REQUIRED_FREE_GB=${REQUIRED_FREE_GB:-70}

export EVAL_DATA_PATH=${EVAL_DATA_PATH:-/data/user/cj/projects/loraplusMSeq/datasets/natural_reasoning_eval}
export EVAL_AFTER_TRAIN=${EVAL_AFTER_TRAIN:-0}
export TEST_MAX_NEW_TOKENS=${TEST_MAX_NEW_TOKENS:-1536}
export TEST_BATCH_SIZE=${TEST_BATCH_SIZE:-2}

bash task_natural_reasoning.sh
