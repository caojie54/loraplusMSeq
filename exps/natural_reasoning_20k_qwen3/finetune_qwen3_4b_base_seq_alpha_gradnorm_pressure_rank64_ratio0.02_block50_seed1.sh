#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

DATASET_PATH=/mnt/petrelfs/caojie1/projects/CoMoL/datasets/natural_reasoning_20k \
EVAL_DATA_PATH=/mnt/petrelfs/caojie1/projects/CoMoL/datasets/natural_reasoning_eval \
OUTPUT_ROOT=/mnt/dhwfile/raise/user/caojie/loraplusMSeq/outputs/natural-reasoning-20k-qwen3-4b-base \
BASE_MODEL_PATH=Qwen/Qwen3-4B-Base \
MAX_LENGTH=1536 \
TRAIN_BATCH_SIZE=4 \
GRADIENT_ACCUMULATION_STEPS=8 \
WARMUP_STEPS=50 \
GRADIENT_CHECKPOINTING=false \
REQUIRED_FREE_GB=75 \
TEST_MAX_NEW_TOKENS=1536 \
TEST_BATCH_SIZE=120 \
NATURAL_REASONING_BENCHMARKS="gpqa_diamond math_500 mmlu_pro_500" \
METHOD=alpha \
ALPHA_SCORE=lora_grad_norm \
RANK=64 \
COMP_RATIO=0.02 \
MODULE_LR=1e-5 \
SELECTION_INTERVAL=50 \
SEED=1 \
NUM_TRAIN_EPOCHS=1 \
LORA_OPTIMIZER_DTYPE=bf16 \
MODULE_OPTIMIZER_DTYPE=fp32 \
MODULE_OPTIMIZER_STATE_STRATEGY=reset_offload \
MODULE_GRADIENT_MODE=full \
SAVE_MERGED_MODEL=true \
RUN_NAME=qwen3-4b-base-seq-alpha-gradnorm-pressure-qkvogateupdown-rank64-natural-reasoning-20k-epoch1-ratio0.02-block50-seed1-loraoptbf16-moduleoptfp32-resetoffload-loralr1e-4-modulelr1e-5 \
bash task_natural_reasoning.sh
