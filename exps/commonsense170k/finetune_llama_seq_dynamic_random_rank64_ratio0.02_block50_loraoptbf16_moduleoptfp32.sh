#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

METHOD=dynamic_random \
RANK=64 \
COMP_RATIO=0.02 \
MODULE_LR=1e-5 \
SELECTION_INTERVAL=50 \
NUM_TRAIN_EPOCHS=1 \
SAVE_MERGED_MODEL=true \
LORA_OPTIMIZER_DTYPE=bf16 \
MODULE_OPTIMIZER_DTYPE=fp32 \
MODULE_GRADIENT_MODE=full \
MODULE_OPTIMIZER_STATE_STRATEGY=reset_offload \
RUN_NAME=llama-3-1-8b-seq-dynamic_random-qkvogateupdown-rank64-commonsense170k-epoch1-ratio0.02-block50-loraoptbf16-moduleoptfp32-resetoffload-loralr1e-4-modulelr1e-5 \
bash task.sh
