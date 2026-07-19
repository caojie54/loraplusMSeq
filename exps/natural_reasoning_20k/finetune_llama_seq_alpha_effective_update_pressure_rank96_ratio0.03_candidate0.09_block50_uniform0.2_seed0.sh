#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

METHOD=alpha \
ALPHA_SCORE=lora_effective_update_pressure \
ALPHA_CANDIDATE_RATIO=0.09 \
ALPHA_SAMPLING_TEMPERATURE=1.0 \
ALPHA_UNIFORM_MIX=0.2 \
ALPHA_SCORE_GAMMA=1.0 \
ALPHA_GROUP_NORM=none \
NUM_TRAIN_EPOCHS=1 \
SAVE_MERGED_MODEL=true \
COMP_RATIO=0.03 \
SELECTION_INTERVAL=50 \
SEED=0 \
LORA_OPTIMIZER_DTYPE=bf16 \
MODULE_OPTIMIZER_DTYPE=fp32 \
MODULE_GRADIENT_MODE=full \
MODULE_OPTIMIZER_STATE_STRATEGY=reset_offload \
RUN_NAME=llama-3-1-8b-seq-alpha-effective-update-pressure-candidate0.09-uniform0.2-qkvogateupdown-rank96-natural-reasoning-20k-epoch1-ratio0.03-block50-seed0-loraoptbf16-moduleoptfp32-resetoffload-loralr1e-4-modulelr1e-5 \
DATASET_PATH=/mnt/petrelfs/caojie1/projects/CoMoL/datasets/natural_reasoning_20k \
EVAL_DATA_PATH=/mnt/petrelfs/caojie1/projects/CoMoL/datasets/natural_reasoning_eval \
OUTPUT_ROOT=/mnt/dhwfile/raise/user/caojie/loraplusMSeq/outputs/natural-reasoning-20k \
RANK=96 \
LORA_LR=1e-4 \
MODULE_LR=1e-5 \
LORA_DROPOUT=0.0 \
MAX_LENGTH=1536 \
TRAIN_BATCH_SIZE=4 \
GRADIENT_ACCUMULATION_STEPS=8 \
WARMUP_STEPS=50 \
GRADIENT_CHECKPOINTING=false \
REQUIRED_FREE_GB=75 \
TEST_MAX_NEW_TOKENS=1536 \
TEST_BATCH_SIZE=64 \
NATURAL_REASONING_BENCHMARKS="gpqa_diamond math_500 mmlu_pro_500" \
bash task_natural_reasoning.sh
