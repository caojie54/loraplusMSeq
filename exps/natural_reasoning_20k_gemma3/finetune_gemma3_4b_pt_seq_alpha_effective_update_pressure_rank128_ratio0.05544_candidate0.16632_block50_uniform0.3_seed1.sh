#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
set -a

BASE_MODEL_PATH=google/gemma-3-4b-pt
OUTPUT_ROOT=/mnt/dhwfile/raise/user/caojie/loraplusMSeq/outputs/natural-reasoning-20k-gemma-3-4b-pt
METHOD=alpha
ALPHA_SCORE=lora_effective_update_pressure
ALPHA_CANDIDATE_RATIO=0.16632
ALPHA_SAMPLING_TEMPERATURE=1.0
ALPHA_UNIFORM_MIX=0.3
ALPHA_SCORE_GAMMA=1.0
ALPHA_GROUP_NORM=none
RANK=128
COMP_RATIO=0.05544
SELECTION_INTERVAL=50
NUM_TRAIN_EPOCHS=1
SEED=1
LORA_OPTIMIZER_DTYPE=bf16
MODULE_OPTIMIZER_DTYPE=fp32
MODULE_OPTIMIZER_STATE_STRATEGY=reset_offload
LORA_LR=1e-4
MODULE_LR=1e-5
RUN_NAME=gemma-3-4b-pt-seq-alpha-effective-update-pressure-candidate0.16632-uniform0.3-qkvogateupdown-rank128-natural-reasoning-20k-epoch1-ratio0.05544-block50-seed1-loraoptbf16-moduleoptfp32-resetoffload-loralr1e-4-modulelr1e-5
MAX_LENGTH=1536
TRAIN_BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=8
GRADIENT_CHECKPOINTING=false
REQUIRED_FREE_GB=75
TEST_MAX_NEW_TOKENS=1536
TEST_BATCH_SIZE=120
NATURAL_REASONING_BENCHMARKS="gpqa_diamond math_500 mmlu_pro_500"

set +a

bash task_natural_reasoning.sh
