#!/usr/bin/env bash
set -euo pipefail

BASE_MODEL_PATH=google/gemma-3-4b-pt \
OUTPUT_ROOT=/mnt/dhwfile/raise/user/caojie/loraplusMSeq/outputs/natural-reasoning-20k-gemma-3-4b-pt \
METHOD=alpha \
ALPHA_SCORE=lora_grad_norm \
RANK=64 \
COMP_RATIO=0.02772 \
SELECTION_INTERVAL=50 \
NUM_TRAIN_EPOCHS=1 \
SEED=1 \
LORA_OPTIMIZER_DTYPE=bf16 \
MODULE_OPTIMIZER_DTYPE=fp32 \
MODULE_OPTIMIZER_STATE_STRATEGY=reset_offload \
LORA_LR=1e-4 \
MODULE_LR=1e-5 \
RUN_NAME=gemma-3-4b-pt-seq-alpha-gradnorm-pressure-qkvogateupdown-rank64-natural-reasoning-20k-epoch1-ratio0.02772-block50-seed1-loraoptbf16-moduleoptfp32-resetoffload-loralr1e-4-modulelr1e-5 \
MAX_LENGTH=1536 \
TRAIN_BATCH_SIZE=4 \
GRADIENT_ACCUMULATION_STEPS=8 \
GRADIENT_CHECKPOINTING=false \
REQUIRED_FREE_GB=75 \
TEST_MAX_NEW_TOKENS=1536 \
TEST_BATCH_SIZE=120 \
bash task_natural_reasoning.sh
