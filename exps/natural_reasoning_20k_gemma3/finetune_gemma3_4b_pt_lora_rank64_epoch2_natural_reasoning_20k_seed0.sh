#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 

BASE_MODEL_PATH=google/gemma-3-4b-pt \
OUTPUT_ROOT=/data/user/cj/experiments/loraplusMSeq/outputs/natural-reasoning-20k/gemma-3-4b-pt \
METHOD=lora \
RANK=64 \
NUM_TRAIN_EPOCHS=2 \
SEED=0 \
LORA_OPTIMIZER_DTYPE=bf16 \
LORA_LR=1e-4 \
SAVE_MERGED_MODEL=false \
RUN_NAME=gemma-3-4b-pt-lora-qkvogateupdown-rank64-natural-reasoning-20k-epoch2-seed0-loraoptbf16-loralr1e-4 \
MAX_LENGTH=1536 \
TRAIN_BATCH_SIZE=4 \
GRADIENT_ACCUMULATION_STEPS=8 \
GRADIENT_CHECKPOINTING=false \
REQUIRED_FREE_GB=75 \
TEST_MAX_NEW_TOKENS=1536 \
TEST_BATCH_SIZE=120 \
bash task_natural_reasoning.sh
