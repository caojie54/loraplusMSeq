#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

METHOD=dynamic_random \
RANK=64 \
COMP_RATIO=0.02 \
MODULE_LR=1e-5 \
SELECTION_INTERVAL=50 \
MODULE_OPTIMIZER_STATE_STRATEGY=persistent_offload \
LORA_OPTIMIZER_RESET_STRATEGY=keep \
RUN_NAME=llama-3-1-8b-seq-dynamic_random-moduleopt-persistent_offload-qkvogateupdown-rank64-commonsense170k-epoch1-ratio0.02-block50-loralr1e-4-modulelr1e-5 \
bash task.sh
