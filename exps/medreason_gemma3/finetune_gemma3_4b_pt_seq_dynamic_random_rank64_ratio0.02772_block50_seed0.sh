#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

# Inherit CUDA_VISIBLE_DEVICES, or scan all GPUs when unset.
export METHOD=dynamic_random
export SAVE_MERGED_MODEL=true
export BASE_MODEL_PATH=${BASE_MODEL_PATH:-google/gemma-3-4b-pt}
export DATASET_PATH=${DATASET_PATH:-/data/user/cj/datasets/MedReason/ours_quality_33000.jsonl}
export OUTPUT_ROOT=${OUTPUT_ROOT:-/data/user/cj/data2/experiments/loraplusMSeq/outputs/medreason/gemma-3-4b-pt}
export RANK=${RANK:-64}
export COMP_RATIO=${COMP_RATIO:-0.02772}
export ALPHA_CANDIDATE_RATIO=${ALPHA_CANDIDATE_RATIO:-0}
export SELECTION_INTERVAL=${SELECTION_INTERVAL:-50}
export NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-1}
export SEED=${SEED:-0}
export LORA_OPTIMIZER_DTYPE=${LORA_OPTIMIZER_DTYPE:-bf16}
export MODULE_OPTIMIZER_DTYPE=${MODULE_OPTIMIZER_DTYPE:-fp32}
export MODULE_OPTIMIZER_STATE_STRATEGY=${MODULE_OPTIMIZER_STATE_STRATEGY:-reset_offload}
export LORA_LR=${LORA_LR:-1e-4}
export MODULE_LR=${MODULE_LR:-1e-5}
export MAX_LENGTH=${MAX_LENGTH:-1200}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-4}
export GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-8}
export GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-false}
export REQUIRED_FREE_GB=${REQUIRED_FREE_GB:-70}

bash task_medreason.sh
