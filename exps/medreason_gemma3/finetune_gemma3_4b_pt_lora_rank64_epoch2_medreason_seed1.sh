#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

# Restrict candidates if needed: CUDA_VISIBLE_DEVICES=1,2 bash <this script>
export BASE_MODEL_PATH=${BASE_MODEL_PATH:-google/gemma-3-4b-pt}
export DATASET_PATH=${DATASET_PATH:-/data/user/cj/datasets/MedReason/ours_quality_33000.jsonl}
export OUTPUT_ROOT=${OUTPUT_ROOT:-/data/user/cj/data2/experiments/loraplusMSeq/outputs/medreason/gemma-3-4b-pt}
export METHOD=lora
export RANK=64
export NUM_TRAIN_EPOCHS=2
export SEED=1
export LORA_OPTIMIZER_DTYPE=bf16
export LORA_LR=1e-4
export SAVE_MERGED_MODEL=true
export RUN_NAME=${RUN_NAME:-gemma-3-4b-pt-lora-qkvogateupdown-rank64-medreason-epoch2-seed1-loraoptbf16-loralr1e-4}
export MAX_LENGTH=${MAX_LENGTH:-1200}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-4}
export GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-8}
export GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-false}
export REQUIRED_FREE_GB=${REQUIRED_FREE_GB:-75}

bash task_medreason.sh
