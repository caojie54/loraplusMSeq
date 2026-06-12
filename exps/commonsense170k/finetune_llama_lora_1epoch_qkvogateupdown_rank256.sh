#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

METHOD=lora \
NUM_TRAIN_EPOCHS=1 \
RANK=256 \
SAVE_MERGED_MODEL=false \
RUN_NAME=llama-3-1-8b-lora-qkvogateupdown-rank256-commonsense170k-epoch1 \
bash task.sh
