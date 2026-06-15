#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

METHOD=alpha \
RANK=64 \
COMP_RATIO=0.02 \
MODULE_LR=1e-5 \
SELECTION_INTERVAL=5326 \
ALPHA_SCORE=lora_grad_norm_min \
LORA_OPTIMIZER_RESET_STRATEGY=keep \
RUN_NAME=llama-3-1-8b-seq-alpha-gradnormmin-qkvogateupdown-rank64-commonsense170k-epoch1-ratio0.02-block5326-loralr1e-4-modulelr1e-5 \
bash task.sh
