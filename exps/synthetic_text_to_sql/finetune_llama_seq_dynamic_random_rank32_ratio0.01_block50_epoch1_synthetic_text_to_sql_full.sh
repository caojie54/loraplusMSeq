#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
set -a

METHOD=dynamic_random
NUM_TRAIN_EPOCHS=1
SAVE_MERGED_MODEL=true
COMP_RATIO=0.01
SELECTION_INTERVAL=50
SEED=0
LORA_OPTIMIZER_DTYPE=bf16
MODULE_OPTIMIZER_DTYPE=fp32
MODULE_GRADIENT_MODE=full
MODULE_OPTIMIZER_STATE_STRATEGY=reset_offload
RUN_NAME=llama-3-1-8b-seq-dynamic_random-qkvogateupdown-rank32-synthetic-text-to-sql-full-outputonly-epoch1-ratio0.01-block50-seed0-loraoptbf16-moduleoptfp32-resetoffload-loralr1e-4-modulelr1e-5
BASE_MODEL_PATH=${BASE_MODEL_PATH:-meta-llama/Llama-3.1-8B}
DATASET_PATH=${DATASET_PATH:-/mnt/petrelfs/caojie1/projects/CoMoL/datasets/synthetic_text_to_sql_full}
EVAL_DATA_PATH=${EVAL_DATA_PATH:-/mnt/petrelfs/caojie1/projects/CoMoL/datasets/synthetic_text_to_sql_eval}
OUTPUT_ROOT=${OUTPUT_ROOT:-/mnt/dhwfile/raise/user/caojie/loraplusMSeq/outputs/synthetic-text-to-sql-full-outputonly}
RANK=32
LORA_LR=1e-4
MODULE_LR=1e-5
LORA_DROPOUT=0.0
MAX_LENGTH=512
TRAIN_BATCH_SIZE=12
GRADIENT_ACCUMULATION_STEPS=2
WARMUP_STEPS=250
GRADIENT_CHECKPOINTING=false
REQUIRED_FREE_GB=75
TEST_MAX_NEW_TOKENS=128
TEST_BATCH_SIZE=416
set +a

bash task_synthetic_text_to_sql.sh
