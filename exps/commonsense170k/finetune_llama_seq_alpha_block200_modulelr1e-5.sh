#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

METHOD=alpha \
MODULE_LR=1e-5 \
SELECTION_INTERVAL=200 \
RUN_NAME=llama-3-1-8b-seq-alpha-qkvogateupdown-rank32-commonsense170k-epoch1-ratio0.005-block200-loralr1e-4-modulelr1e-5 \
bash task.sh
