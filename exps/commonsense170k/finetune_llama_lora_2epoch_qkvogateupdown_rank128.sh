#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

METHOD=lora \
NUM_TRAIN_EPOCHS=2 \
RANK=128 \
SAVE_MERGED_MODEL=false \
RUN_NAME=llama-3-1-8b-lora-qkvogateupdown-rank128-commonsense170k-epoch2 \
bash task.sh
