#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

METHOD=static_random \
MODULE_LR=1e-5 \
SELECTION_INTERVAL=100 \
RUN_NAME=llama-3-1-8b-seq-static_random-qkvogateupdown-rank32-commonsense170k-epoch1-ratio0.005-block100-loralr1e-4-modulelr1e-5 \
bash task.sh
