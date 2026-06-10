#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
METHOD=lora NUM_TRAIN_EPOCHS=2 RUN_NAME=llama-3-1-8b-lora-qkvogateupdown-rank32-commonsense170k-epoch2 bash task.sh

