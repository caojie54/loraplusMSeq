#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
set -a

METHOD=alpha
ALPHA_SCORE=lora_effective_update_pressure
ALPHA_CANDIDATE_RATIO=0.06
ALPHA_UNIFORM_MIX=0.1
RANK=64
LORA_DROPOUT=0.1
COMP_RATIO=0.02
LORA_LR=1e-4
MODULE_LR=1e-5
SELECTION_INTERVAL=50
NUM_TRAIN_EPOCHS=1
SAVE_MERGED_MODEL=true
LORA_OPTIMIZER_DTYPE=bf16
MODULE_OPTIMIZER_DTYPE=fp32
MODULE_GRADIENT_MODE=full
MODULE_OPTIMIZER_STATE_STRATEGY=reset_offload
REQUIRED_FREE_GB=75
RUN_NAME=llama-3-1-8b-seq-alpha-effective-update-pressure-candidate0.06-uniform0.1-dropout0.1-qkvogateupdown-rank64-commonsense170k-epoch1-ratio0.02-block50-loraoptbf16-moduleoptfp32-resetoffload-loralr1e-4-modulelr1e-5

set +a

bash task_commonsense.sh
