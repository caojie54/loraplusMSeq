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
METHOD=alpha
ALPHA_SCORE=lora_grad_norm
RANK=64
COMP_RATIO=0.02772
LORA_LR=1e-4
MODULE_LR=1e-5
SELECTION_INTERVAL=500
NUM_TRAIN_EPOCHS=1
SAVE_MERGED_MODEL=true
LORA_OPTIMIZER_DTYPE=bf16
MODULE_OPTIMIZER_DTYPE=fp32
MODULE_GRADIENT_MODE=full
MODULE_OPTIMIZER_STATE_STRATEGY=reset_offload
RUN_NAME=gemma-3-4b-pt-seq-alpha-gradnorm-pressure-qkvogateupdown-rank64-commonsense170k-epoch1-ratio0.02772-block500-loraoptbf16-moduleoptfp32-resetoffload-loralr1e-4-modulelr1e-5

set +a

bash task_commonsense.sh
