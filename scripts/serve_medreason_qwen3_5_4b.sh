#!/usr/bin/env bash
set -euo pipefail
# CUDA_VISIBLE_DEVICES is inherited from wait_for_gpus.py.
LOG_NUM=${LOG_NUM:-0}
MODEL_PATH=${MODEL_PATH:-/data/user/cj/data2/experiments/loraplusMSeq/outputs/medreason/Qwen3-5-4B-Base/qwen3.5-4b-base-lora-qkvogateupdown-rank64-medreason-epoch2-seed0-loraoptbf16-loralr1e-4/merged}
PORT=${PORT:-28${LOG_NUM}35}
LOG_FILE=${LOG_FILE:-sglang${LOG_NUM}.log}

if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
  echo "Missing model config: ${MODEL_PATH}/config.json" >&2
  exit 2
fi
echo "Starting SGLang on GPU(s) ${CUDA_VISIBLE_DEVICES:-unset}; port=${PORT}; log=${LOG_FILE}"
# Keep the server in the foreground; background the entire waiting command instead.
exec python -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --port "${PORT}" \
  --mem-fraction-static "${MEM_FRACTION_STATIC:-0.8}" \
  --dp "${DP_SIZE:-1}" \
  --tp "${TP_SIZE:-1}" > "${LOG_FILE}" 2>&1
