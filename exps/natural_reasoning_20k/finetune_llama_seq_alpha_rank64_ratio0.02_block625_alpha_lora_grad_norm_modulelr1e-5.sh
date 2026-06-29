#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

METHOD=alpha \
ALPHA_SCORE=lora_grad_norm \
DATASET_PATH=/mnt/petrelfs/caojie1/projects/CoMoL/datasets/natural_reasoning_20k \
EVAL_DATA_PATH=/mnt/petrelfs/caojie1/projects/CoMoL/datasets/natural_reasoning_eval \
OUTPUT_ROOT=/mnt/dhwfile/raise/user/caojie/loraplusMSeq/outputs/natural-reasoning-20k \
RANK=64 \
LORA_LR=1e-4 \
MODULE_LR=1e-5 \
LORA_DROPOUT=0.0 \
COMP_RATIO=0.02 \
SELECTION_INTERVAL=625 \
MODULE_OPTIMIZER_STATE_STRATEGY=reset_offload \
NUM_TRAIN_EPOCHS=1 \
SAVE_MERGED_MODEL=true \
MAX_LENGTH=1536 \
TRAIN_BATCH_SIZE=4 \
GRADIENT_ACCUMULATION_STEPS=8 \
WARMUP_STEPS=50 \
GRADIENT_CHECKPOINTING=false \
REQUIRED_FREE_GB=75 \
TEST_MAX_NEW_TOKENS=1536 \
TEST_BATCH_SIZE=64 \
NATURAL_REASONING_BENCHMARKS="gpqa_diamond math_500 mmlu_pro_500" \
RUN_NAME=llama-3-1-8b-seq-alpha-gradnorm-qkvogateupdown-rank64-natural-reasoning-20k-epoch1-ratio0.02-block625-loralr1e-4-modulelr1e-5 \
bash task_natural_reasoning.sh
