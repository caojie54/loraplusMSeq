#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

METHOD=lora \
BASE_MODEL_PATH=google/gemma-3-1b-pt \
DATASET_PATH=/mnt/petrelfs/caojie1/projects/CoMoL/datasets/natural_reasoning_20k \
EVAL_DATA_PATH=/mnt/petrelfs/caojie1/projects/CoMoL/datasets/natural_reasoning_eval \
OUTPUT_ROOT=/mnt/dhwfile/raise/user/caojie/loraplusMSeq/outputs/natural-reasoning-20k-gemma-3-1b-pt \
RANK=128 \
SEED=0 \
NUM_TRAIN_EPOCHS=2 \
SAVE_MERGED_MODEL=false \
LORA_OPTIMIZER_DTYPE=bf16 \
MODULE_OPTIMIZER_DTYPE=bf16 \
MODULE_GRADIENT_MODE=full \
MODULE_OPTIMIZER_STATE_STRATEGY=reset_offload \
RUN_NAME=gemma-3-1b-pt-lora-qkvogateupdown-rank128-natural-reasoning-20k-epoch2-seed0-loraoptbf16-loralr1e-4 \
MAX_LENGTH=1536 \
TRAIN_BATCH_SIZE=8 \
GRADIENT_ACCUMULATION_STEPS=4 \
WARMUP_STEPS=50 \
GRADIENT_CHECKPOINTING=false \
REQUIRED_FREE_GB=75 \
TEST_MAX_NEW_TOKENS=1536 \
TEST_BATCH_SIZE=256 \
NATURAL_REASONING_BENCHMARKS="gpqa_diamond math_500 mmlu_pro_500" \
bash task_natural_reasoning.sh
