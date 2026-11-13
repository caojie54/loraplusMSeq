#!/usr/bin/env bash
set -euo pipefail

# Start SGLang on MODEL_DIR first; --model_name in eval.py does not load weights.
: "${MODEL_DIR:?Set MODEL_DIR to the absolute merged/ path of the trained run}"
[[ -f "${MODEL_DIR}/config.json" ]] || { echo "Missing model config: ${MODEL_DIR}" >&2; exit 2; }
MEDREASON_ROOT=${MEDREASON_ROOT:-/data/user/cj/projects/MedReason}
PORT=${PORT:-28035}
RUN_DIR=$(dirname "${MODEL_DIR%/}")
TASK_FOLDER=${TASK_FOLDER:-Med_eval/$(basename "${RUN_DIR}")}
cd "${MEDREASON_ROOT}"
mkdir -p "results/${TASK_FOLDER}/logs" "results/${TASK_FOLDER}/result"

for dataset in medbullets_op4 medbullets_op5 HLE_biomed medqa_test MedXpertQA_test; do
  python ./src/evaluation/eval.py \
    --model_name "${MODEL_DIR}" \
    --eval_file "./eval_data/${dataset}.jsonl" \
    --port "${PORT}" \
    --strict_prompt \
    --no-use_chat_template \
    --batch_size "${TEST_BATCH_SIZE:-1000}" \
    --max_new_tokens "${TEST_MAX_NEW_TOKENS:-2000}" \
    --task_floder "${TASK_FOLDER}"
done
