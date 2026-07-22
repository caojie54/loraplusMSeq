#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

BASE_MODEL_PATH=google/gemma-3-1b-pt \
OUTPUT_ROOT=/mnt/dhwfile/raise/user/caojie/loraplusMSeq/outputs/medmcqa-full-outputonly-gemma-3-1b-pt \
METHOD=alpha \
ALPHA_SCORE=lora_effective_update_pressure \
ALPHA_CANDIDATE_RATIO=0.078285 \
ALPHA_SAMPLING_TEMPERATURE=1.0 \
ALPHA_UNIFORM_MIX=0.1 \
ALPHA_SCORE_GAMMA=1.0 \
ALPHA_GROUP_NORM=none \
RANK=32 \
COMP_RATIO=0.026095 \
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
RUN_NAME=gemma-3-1b-pt-seq-alpha-effective-update-pressure-candidate0.078285-uniform0.1-qkvogateupdown-rank32-medmcqa-full-outputonly-epoch1-ratio0.026095-block50-seed0-loraoptbf16-moduleoptfp32-resetoffload-loralr1e-4-modulelr1e-5 \
DATASET_PATH=/mnt/petrelfs/caojie1/projects/CoMoL/datasets/medmcqa_full \
EVAL_DATA_PATH=/mnt/petrelfs/caojie1/projects/CoMoL/datasets/medmcqa_eval \
MAX_LENGTH=256 \
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-48} \
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1} \
WARMUP_STEPS=400 \
GRADIENT_CHECKPOINTING=false \
REQUIRED_FREE_GB=75 \
TEST_MAX_NEW_TOKENS=8 \
TEST_BATCH_SIZE=${TEST_BATCH_SIZE:-768} \
bash task_medmcqa.sh
