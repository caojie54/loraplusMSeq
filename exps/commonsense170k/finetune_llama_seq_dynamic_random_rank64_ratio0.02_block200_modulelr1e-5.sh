#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

METHOD=dynamic_random \
RANK=64 \
COMP_RATIO=0.02 \
MODULE_LR=1e-5 \
SELECTION_INTERVAL=200 \
RUN_NAME=llama-3-1-8b-seq-dynamic_random-qkvogateupdown-rank64-commonsense170k-epoch1-ratio0.02-block200-loralr1e-4-modulelr1e-5 \
bash task.sh
