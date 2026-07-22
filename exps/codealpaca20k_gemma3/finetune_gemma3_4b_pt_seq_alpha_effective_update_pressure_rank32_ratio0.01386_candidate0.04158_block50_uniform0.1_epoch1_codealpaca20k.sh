#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

BASE_MODEL_PATH=google/gemma-3-4b-pt \
OUTPUT_ROOT=/mnt/dhwfile/raise/user/caojie/loraplusMSeq/outputs/codealpaca20k-gemma-3-4b-pt \
METHOD=alpha \
ALPHA_SCORE=lora_effective_update_pressure \
ALPHA_CANDIDATE_RATIO=0.04158 \
ALPHA_SAMPLING_TEMPERATURE=1.0 \
ALPHA_UNIFORM_MIX=0.1 \
ALPHA_SCORE_GAMMA=1.0 \
ALPHA_GROUP_NORM=none \
RANK=32 \
COMP_RATIO=0.01386 \
SELECTION_INTERVAL=50 \
NUM_TRAIN_EPOCHS=1 \
SEED=0 \
LORA_OPTIMIZER_DTYPE=bf16 \
MODULE_OPTIMIZER_DTYPE=fp32 \
MODULE_OPTIMIZER_STATE_STRATEGY=reset_offload \
MODULE_GRADIENT_MODE=full \
SAVE_MERGED_MODEL=true \
LORA_LR=1e-4 \
MODULE_LR=1e-5 \
RUN_NAME=gemma-3-4b-pt-seq-alpha-effective-update-pressure-candidate0.04158-uniform0.1-qkvogateupdown-rank32-codealpaca20k-epoch1-ratio0.01386-block50-seed0-loraoptbf16-moduleoptfp32-resetoffload-loralr1e-4-modulelr1e-5 \
DATASET_PATH=/mnt/petrelfs/caojie1/projects/CoMoL/datasets/codealpaca20k \
EVAL_DATA_PATH=/mnt/petrelfs/caojie1/projects/CoMoL/datasets/eval_code \
MAX_LENGTH=500 \
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-12} \
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1} \
WARMUP_STEPS=200 \
GRADIENT_CHECKPOINTING=false \
REQUIRED_FREE_GB=75 \
TEST_MAX_NEW_TOKENS=400 \
TEST_BATCH_SIZE=${TEST_BATCH_SIZE:-48} \
bash task_codealpaca20k.sh
