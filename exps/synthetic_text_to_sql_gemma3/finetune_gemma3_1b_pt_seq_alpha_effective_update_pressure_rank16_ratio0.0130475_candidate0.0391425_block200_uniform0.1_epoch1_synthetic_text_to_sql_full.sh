#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

BASE_MODEL_PATH=google/gemma-3-1b-pt \
OUTPUT_ROOT=/mnt/dhwfile/raise/user/caojie/loraplusMSeq/outputs/synthetic-text-to-sql-full-outputonly-gemma-3-1b-pt \
METHOD=alpha \
ALPHA_SCORE=lora_effective_update_pressure \
ALPHA_CANDIDATE_RATIO=0.0391425 \
ALPHA_SAMPLING_TEMPERATURE=1.0 \
ALPHA_UNIFORM_MIX=0.1 \
ALPHA_SCORE_GAMMA=1.0 \
ALPHA_GROUP_NORM=none \
RANK=16 \
COMP_RATIO=0.0130475 \
SELECTION_INTERVAL=200 \
NUM_TRAIN_EPOCHS=1 \
SEED=0 \
LORA_OPTIMIZER_DTYPE=bf16 \
MODULE_OPTIMIZER_DTYPE=fp32 \
MODULE_OPTIMIZER_STATE_STRATEGY=reset_offload \
MODULE_GRADIENT_MODE=full \
SAVE_MERGED_MODEL=true \
LORA_LR=1e-4 \
MODULE_LR=1e-5 \
RUN_NAME=gemma-3-1b-pt-seq-alpha-effective-update-pressure-candidate0.0391425-uniform0.1-qkvogateupdown-rank16-synthetic-text-to-sql-full-outputonly-epoch1-ratio0.0130475-block200-seed0-loraoptbf16-moduleoptfp32-resetoffload-loralr1e-4-modulelr1e-5 \
DATASET_PATH=/mnt/petrelfs/caojie1/projects/CoMoL/datasets/synthetic_text_to_sql_full \
EVAL_DATA_PATH=/mnt/petrelfs/caojie1/projects/CoMoL/datasets/synthetic_text_to_sql_eval \
MAX_LENGTH=512 \
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-24} \
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1} \
WARMUP_STEPS=250 \
GRADIENT_CHECKPOINTING=false \
REQUIRED_FREE_GB=75 \
TEST_MAX_NEW_TOKENS=128 \
TEST_BATCH_SIZE=${TEST_BATCH_SIZE:-800} \
bash task_synthetic_text_to_sql.sh
