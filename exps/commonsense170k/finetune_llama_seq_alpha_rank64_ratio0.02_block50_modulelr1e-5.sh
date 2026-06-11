#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

METHOD=alpha \
RANK=64 \
COMP_RATIO=0.02 \
MODULE_LR=1e-5 \
SELECTION_INTERVAL=50 \
RUN_NAME=llama-3-1-8b-seq-alpha-qkvogateupdown-rank64-commonsense170k-epoch1-ratio0.02-block50-loralr1e-4-modulelr1e-5 \
bash task.sh
