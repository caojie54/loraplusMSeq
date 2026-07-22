#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
set -a

METHOD=${METHOD:-lora}
NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-2}
SAVE_MERGED_MODEL=${SAVE_MERGED_MODEL:-false}
BASE_MODEL_PATH=${BASE_MODEL_PATH:-meta-llama/Llama-3.1-8B}
DATASET_PATH=${DATASET_PATH:-/mnt/petrelfs/caojie1/projects/CoMoL/datasets/medmcqa_full}
EVAL_DATA_PATH=${EVAL_DATA_PATH:-/mnt/petrelfs/caojie1/projects/CoMoL/datasets/medmcqa_eval}
OUTPUT_ROOT=${OUTPUT_ROOT:-/mnt/dhwfile/raise/user/caojie/loraplusMSeq/outputs/medmcqa-full-outputonly}
RUN_NAME=${RUN_NAME:-llama-3-1-8b-lora-qkvogateupdown-rank32-medmcqa-full-outputonly-epoch2-loraoptbf16-loralr1e-4}
RANK=${RANK:-32}
SEED=${SEED:-0}
LORA_LR=${LORA_LR:-1e-4}
MODULE_LR=${MODULE_LR:-1e-5}
LORA_DROPOUT=${LORA_DROPOUT:-0.0}
LORA_OPTIMIZER_DTYPE=${LORA_OPTIMIZER_DTYPE:-bf16}
MODULE_OPTIMIZER_DTYPE=${MODULE_OPTIMIZER_DTYPE:-bf16}
MAX_LENGTH=${MAX_LENGTH:-256}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-24}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1}
WARMUP_STEPS=${WARMUP_STEPS:-400}
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-false}
REQUIRED_FREE_GB=${REQUIRED_FREE_GB:-75}
TEST_MAX_NEW_TOKENS=${TEST_MAX_NEW_TOKENS:-8}
TEST_BATCH_SIZE=${TEST_BATCH_SIZE:-800}
set +a

bash task_medmcqa.sh
