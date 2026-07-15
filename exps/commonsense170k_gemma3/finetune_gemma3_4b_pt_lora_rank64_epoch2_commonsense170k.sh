#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
set -a

BASE_MODEL_PATH=google/gemma-3-4b-pt
OUTPUT_ROOT=/mnt/dhwfile/raise/user/caojie/loraplusMSeq/outputs/commonsense170k-gemma-3-4b-pt
MAX_LENGTH=256
TRAIN_BATCH_SIZE=16
GRADIENT_ACCUMULATION_STEPS=2
REQUIRED_FREE_GB=75
TEST_MAX_NEW_TOKENS=64
TEST_BATCH_SIZE=256
METHOD=lora
RANK=64
LORA_LR=1e-4
NUM_TRAIN_EPOCHS=2
SAVE_MERGED_MODEL=false
LORA_OPTIMIZER_DTYPE=bf16
RUN_NAME=gemma-3-4b-pt-lora-qkvogateupdown-rank64-commonsense170k-epoch2-loraoptbf16-loralr1e-4

set +a

bash task_commonsense.sh
