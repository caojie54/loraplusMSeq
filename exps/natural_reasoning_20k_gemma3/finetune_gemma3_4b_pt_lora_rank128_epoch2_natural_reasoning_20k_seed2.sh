#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
set -a

BASE_MODEL_PATH=google/gemma-3-4b-pt
OUTPUT_ROOT=/mnt/dhwfile/raise/user/caojie/loraplusMSeq/outputs/natural-reasoning-20k-gemma-3-4b-pt
METHOD=lora
RANK=128
NUM_TRAIN_EPOCHS=2
SEED=2
LORA_OPTIMIZER_DTYPE=bf16
LORA_LR=1e-4
SAVE_MERGED_MODEL=false
RUN_NAME=gemma-3-4b-pt-lora-qkvogateupdown-rank128-natural-reasoning-20k-epoch2-seed2-loraoptbf16-loralr1e-4
MAX_LENGTH=1536
TRAIN_BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=8
GRADIENT_CHECKPOINTING=false
REQUIRED_FREE_GB=75
TEST_MAX_NEW_TOKENS=1536
TEST_BATCH_SIZE=120
NATURAL_REASONING_BENCHMARKS="gpqa_diamond math_500 mmlu_pro_500"

set +a

bash task_natural_reasoning.sh
