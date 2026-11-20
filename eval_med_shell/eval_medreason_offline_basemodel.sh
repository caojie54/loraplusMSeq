#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=4

MEDREASON_ROOT=${MEDREASON_ROOT:-/data/user/cj/projects/loraplusMSeq/MedReason}

MODEL_DIR=${MODEL_DIR:-google/gemma-4-12B-it}

MODEL_NAME=$(basename "${MODEL_DIR%/}")
TASK_FOLDER=${TASK_FOLDER:-Med_eval/${MODEL_NAME}-offline}
RESULTS_DIR=${RESULTS_DIR:-/data/user/cj/data2/MedReason}

cd "${MEDREASON_ROOT}"

python ./evaluation/eval_sglang_offline.py \
    --model_name "${MODEL_DIR}" \
    --strict_prompt \
    --no-use_chat_template \
    --batch_size "${TEST_BATCH_SIZE:-400}" \
    --max_new_tokens "${TEST_MAX_NEW_TOKENS:-2000}" \
    --task_floder "${TASK_FOLDER}" \
    --results_dir "${RESULTS_DIR}" \
    --tp_size=1 \
    --dp_size=1 \
    --mem_fraction_static 0.8 \
