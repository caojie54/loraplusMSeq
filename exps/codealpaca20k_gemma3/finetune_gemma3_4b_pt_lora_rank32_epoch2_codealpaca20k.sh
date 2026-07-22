#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

BASE_MODEL_PATH=google/gemma-3-4b-pt \
OUTPUT_ROOT=/mnt/dhwfile/raise/user/caojie/loraplusMSeq/outputs/codealpaca20k-gemma-3-4b-pt \
METHOD=lora \
RANK=32 \
NUM_TRAIN_EPOCHS=2 \
SEED=0 \
LORA_OPTIMIZER_DTYPE=bf16 \
LORA_LR=1e-4 \
SAVE_MERGED_MODEL=false \
RUN_NAME=gemma-3-4b-pt-lora-qkvogateupdown-rank32-codealpaca20k-epoch2-seed0-loraoptbf16-loralr1e-4 \
DATASET_PATH=/mnt/petrelfs/caojie1/projects/CoMoL/datasets/codealpaca20k \
EVAL_DATA_PATH=/mnt/petrelfs/caojie1/projects/CoMoL/datasets/eval_code \
MAX_LENGTH=500 \
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-12} \
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1} \
WARMUP_STEPS=200 \
GRADIENT_CHECKPOINTING=false \
REQUIRED_FREE_GB=75 \
TEST_MAX_NEW_TOKENS=400 \
TEST_BATCH_SIZE=${TEST_BATCH_SIZE:-48} \
bash task_codealpaca20k.sh
