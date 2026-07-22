#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

source /mnt/petrelfs/caojie1/anaconda3/etc/profile.d/conda.sh
conda activate loraplusm

export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export HF_HOME=${HF_HOME:-/mnt/dhwfile/raise/user/caojie/huggingface}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}

MODEL_KEY=${MODEL_KEY:?Set MODEL_KEY}
case "${MODEL_KEY}" in
  gemma3_1b_pt)
    BASE_MODEL_PATH=${BASE_MODEL_PATH:-google/gemma-3-1b-pt}
    MODEL_SLUG=gemma-3-1b-pt
    DEFAULT_TEST_BATCH_SIZE=164
    ;;
  gemma3_1b_it)
    BASE_MODEL_PATH=${BASE_MODEL_PATH:-google/gemma-3-1b-it}
    MODEL_SLUG=gemma-3-1b-it
    DEFAULT_TEST_BATCH_SIZE=164
    ;;
  gemma3_4b_pt)
    BASE_MODEL_PATH=${BASE_MODEL_PATH:-google/gemma-3-4b-pt}
    MODEL_SLUG=gemma-3-4b-pt
    DEFAULT_TEST_BATCH_SIZE=48
    ;;
  gemma3_4b_it)
    BASE_MODEL_PATH=${BASE_MODEL_PATH:-google/gemma-3-4b-it}
    MODEL_SLUG=gemma-3-4b-it
    DEFAULT_TEST_BATCH_SIZE=48
    ;;
  *)
    echo "Unknown MODEL_KEY=${MODEL_KEY}" >&2
    exit 1
    ;;
esac

EVAL_DATA_PATH=${EVAL_DATA_PATH:-/mnt/petrelfs/caojie1/projects/CoMoL/datasets/eval_code}
OUTPUT_ROOT=${OUTPUT_ROOT:-/mnt/dhwfile/raise/user/caojie/loraplusMSeq/outputs/codealpaca20k-basemodel-humaneval}
OUTPUT_DIR="${OUTPUT_ROOT}/${MODEL_SLUG}"
TEST_MAX_NEW_TOKENS=${TEST_MAX_NEW_TOKENS:-400}
TEST_BATCH_SIZE=${TEST_BATCH_SIZE:-${DEFAULT_TEST_BATCH_SIZE}}
TEST_NUM_RETURN_SEQUENCES=${TEST_NUM_RETURN_SEQUENCES:-10}
GPU_MEMORY_MONITOR=${GPU_MEMORY_MONITOR:-1}
GPU_MONITOR_DIR=${GPU_MONITOR_DIR:-${OUTPUT_ROOT}/gpu_memory}

gpu_monitor_pid=""
gpu_monitor_file=""

start_gpu_monitor() {
  if [[ "${GPU_MEMORY_MONITOR}" != "1" ]] || ! command -v nvidia-smi >/dev/null 2>&1; then
    return
  fi
  mkdir -p "${GPU_MONITOR_DIR}"
  local gpu_selector=${CUDA_VISIBLE_DEVICES:-0}
  gpu_selector=${gpu_selector%%,*}
  gpu_monitor_file="${GPU_MONITOR_DIR}/${SLURM_JOB_ID:-$$}_${MODEL_SLUG}_test.mib"
  nvidia-smi -i "${gpu_selector}" --query-gpu=memory.used --format=csv,noheader,nounits -lms 200 > "${gpu_monitor_file}" &
  gpu_monitor_pid=$!
}

stop_gpu_monitor() {
  if [[ -z "${gpu_monitor_pid}" ]]; then
    return
  fi
  kill "${gpu_monitor_pid}" 2>/dev/null || true
  wait "${gpu_monitor_pid}" 2>/dev/null || true
  local peak
  peak=$(awk 'BEGIN { max = 0 } /^[0-9]+/ { if ($1 > max) max = $1 } END { print max }' "${gpu_monitor_file}")
  echo "GPU_MEMORY_PEAK_MIB=${peak} monitor=${gpu_monitor_file}"
  gpu_monitor_pid=""
}

trap stop_gpu_monitor EXIT

start_gpu_monitor
python test_code10.py \
  --model_path="${BASE_MODEL_PATH}" \
  --data_path="${EVAL_DATA_PATH}" \
  --max_new_tokens="${TEST_MAX_NEW_TOKENS}" \
  --batch_size="${TEST_BATCH_SIZE}" \
  --num_return_sequences="${TEST_NUM_RETURN_SEQUENCES}" \
  --output_dir="${OUTPUT_DIR}"
stop_gpu_monitor

python evaluate_code.py \
  --predict_file "${OUTPUT_DIR}/predictions/humaneval_responses.jsonl"
