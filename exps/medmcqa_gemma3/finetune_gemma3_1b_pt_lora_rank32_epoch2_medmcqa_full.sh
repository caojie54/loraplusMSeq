#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

BASE_MODEL_PATH=google/gemma-3-1b-pt \
OUTPUT_ROOT=/mnt/dhwfile/raise/user/caojie/loraplusMSeq/outputs/medmcqa-full-outputonly-gemma-3-1b-pt \
METHOD=lora \
RANK=32 \
NUM_TRAIN_EPOCHS=2 \
SEED=0 \
LORA_OPTIMIZER_DTYPE=bf16 \
LORA_LR=1e-4 \
SAVE_MERGED_MODEL=false \
RUN_NAME=gemma-3-1b-pt-lora-qkvogateupdown-rank32-medmcqa-full-outputonly-epoch2-seed0-loraoptbf16-loralr1e-4 \
DATASET_PATH=/mnt/petrelfs/caojie1/projects/CoMoL/datasets/medmcqa_full \
EVAL_DATA_PATH=/mnt/petrelfs/caojie1/projects/CoMoL/datasets/medmcqa_eval \
MAX_LENGTH=256 \
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-48} \
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1} \
WARMUP_STEPS=400 \
GRADIENT_CHECKPOINTING=false \
REQUIRED_FREE_GB=75 \
TEST_MAX_NEW_TOKENS=8 \
TEST_BATCH_SIZE=${TEST_BATCH_SIZE:-768} \
bash task_medmcqa.sh
