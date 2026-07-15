#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
set -a

METHOD=lora
NUM_TRAIN_EPOCHS=2
SAVE_MERGED_MODEL=false
LORA_OPTIMIZER_DTYPE=bf16
MODULE_OPTIMIZER_DTYPE=bf16
MODULE_GRADIENT_MODE=full
MODULE_OPTIMIZER_STATE_STRATEGY=reset_offload
RUN_NAME=llama-3-1-8b-lora-qkvogateupdown-rank32-natural-reasoning-20k-epoch2-loraoptbf16-loralr1e-4
DATASET_PATH=/mnt/petrelfs/caojie1/projects/CoMoL/datasets/natural_reasoning_20k
EVAL_DATA_PATH=/mnt/petrelfs/caojie1/projects/CoMoL/datasets/natural_reasoning_eval
OUTPUT_ROOT=/mnt/dhwfile/raise/user/caojie/loraplusMSeq/outputs/natural-reasoning-20k
RANK=32
SEED=0
LORA_LR=1e-4
MODULE_LR=1e-5
LORA_DROPOUT=0.0
MAX_LENGTH=1536
TRAIN_BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=8
WARMUP_STEPS=50
GRADIENT_CHECKPOINTING=false
REQUIRED_FREE_GB=75
TEST_MAX_NEW_TOKENS=1536
TEST_BATCH_SIZE=64
NATURAL_REASONING_BENCHMARKS="gpqa_diamond math_500 mmlu_pro_500"

set +a

bash task_natural_reasoning.sh
