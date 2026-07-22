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

METHOD=${METHOD:-lora}
BASE_MODEL_PATH=${BASE_MODEL_PATH:-google/gemma-3-1b-pt}
DATASET_PATH=${DATASET_PATH:-/mnt/petrelfs/caojie1/projects/CoMoL/datasets/codealpaca20k}
EVAL_DATA_PATH=${EVAL_DATA_PATH:-/mnt/petrelfs/caojie1/projects/CoMoL/datasets/eval_code}
OUTPUT_ROOT=${OUTPUT_ROOT:-/mnt/dhwfile/raise/user/caojie/loraplusMSeq/outputs/codealpaca20k-gemma-3-1b-pt}
RANK=${RANK:-32}
LORA_LR=${LORA_LR:-1e-4}
MODULE_LR=${MODULE_LR:-1e-5}
LORA_DROPOUT=${LORA_DROPOUT:-0.0}
SEED=${SEED:-0}
COMP_RATIO=${COMP_RATIO:-0.026095}
SELECTION_INTERVAL=${SELECTION_INTERVAL:-50}
ALPHA_SCORE=${ALPHA_SCORE:-lora_grad_norm}
ALPHA_CANDIDATE_RATIO=${ALPHA_CANDIDATE_RATIO:-0}
ALPHA_SAMPLING_TEMPERATURE=${ALPHA_SAMPLING_TEMPERATURE:-1.0}
ALPHA_UNIFORM_MIX=${ALPHA_UNIFORM_MIX:-0.1}
ALPHA_SCORE_GAMMA=${ALPHA_SCORE_GAMMA:-1.0}
ALPHA_GROUP_NORM=${ALPHA_GROUP_NORM:-none}
LORA_OPTIMIZER_RESET_STRATEGY=${LORA_OPTIMIZER_RESET_STRATEGY:-keep}
LORA_OPTIMIZER_DTYPE=${LORA_OPTIMIZER_DTYPE:-bf16}
MODULE_OPTIMIZER_DTYPE=${MODULE_OPTIMIZER_DTYPE:-bf16}
MODULE_GRADIENT_MODE=${MODULE_GRADIENT_MODE:-full}
RESIDUAL_RTOL=${RESIDUAL_RTOL:-1e-4}
MODULE_OPTIMIZER_STATE_STRATEGY=${MODULE_OPTIMIZER_STATE_STRATEGY:-reset_offload}

if [[ "${METHOD}" == "lora" ]]; then
  NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-2}
  SAVE_MERGED_MODEL=${SAVE_MERGED_MODEL:-false}
else
  NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-1}
  SAVE_MERGED_MODEL=${SAVE_MERGED_MODEL:-true}
fi

RUN_NAME=${RUN_NAME:-gemma-3-1b-pt-${METHOD}-qkvogateupdown-rank${RANK}-codealpaca20k-epoch${NUM_TRAIN_EPOCHS}-seed${SEED}-loraopt${LORA_OPTIMIZER_DTYPE}-loralr${LORA_LR}}
EVAL_AFTER_TRAIN=${EVAL_AFTER_TRAIN:-1}
MAX_LENGTH=${MAX_LENGTH:-500}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1}
WARMUP_STEPS=${WARMUP_STEPS:-200}
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-false}
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-4}
MAX_TRAIN_SAMPLES=${MAX_TRAIN_SAMPLES:-0}
MAX_EVAL_SAMPLES=${MAX_EVAL_SAMPLES:-0}
REQUIRED_FREE_GB=${REQUIRED_FREE_GB:-75}
GPU_WAIT_INTERVAL_SECONDS=${GPU_WAIT_INTERVAL_SECONDS:-60}
GPU_WAIT_TIMEOUT_SECONDS=${GPU_WAIT_TIMEOUT_SECONDS:-0}
TEST_MAX_NEW_TOKENS=${TEST_MAX_NEW_TOKENS:-400}
TEST_BATCH_SIZE=${TEST_BATCH_SIZE:-48}
TEST_NUM_RETURN_SEQUENCES=${TEST_NUM_RETURN_SEQUENCES:-10}
GPU_MEMORY_MONITOR=${GPU_MEMORY_MONITOR:-1}
GPU_MONITOR_DIR=${GPU_MONITOR_DIR:-${OUTPUT_ROOT}/gpu_memory}
MODEL_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
TRAIN_SUCCESS_MARKER="${MODEL_DIR}/TRAIN_SUCCESS"

gpu_monitor_pid=""
gpu_monitor_file=""

start_gpu_monitor() {
  local phase=$1
  if [[ "${GPU_MEMORY_MONITOR}" != "1" ]] || ! command -v nvidia-smi >/dev/null 2>&1; then
    return
  fi
  mkdir -p "${GPU_MONITOR_DIR}"
  local gpu_selector=${CUDA_VISIBLE_DEVICES:-0}
  gpu_selector=${gpu_selector%%,*}
  gpu_monitor_file="${GPU_MONITOR_DIR}/${SLURM_JOB_ID:-$$}_${phase}.mib"
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

if [[ "${REQUIRED_FREE_GB}" != "0" ]]; then
  echo "Initial CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
  echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-}"
  nvidia-smi || true
  allocated_gpu="${CUDA_VISIBLE_DEVICES:-${SLURM_JOB_GPUS:-}}"
  allocated_gpu="${allocated_gpu%%,*}"
  allocated_gpu="${allocated_gpu//[[:space:]]/}"
  if [[ "${allocated_gpu}" =~ ^[0-9]+$ ]]; then
    query_gpu="${allocated_gpu}"
  else
    query_gpu=0
  fi
  gpu_wait_start=${SECONDS}
  while true; do
    IFS=, read -r gpu_id total_mem used_mem < <(
      nvidia-smi -i "${query_gpu}" --query-gpu=index,memory.total,memory.used --format=csv,noheader,nounits
    )
    gpu_id="${gpu_id//[[:space:]]/}"
    total_mem="${total_mem//[[:space:]]/}"
    used_mem="${used_mem//[[:space:]]/}"
    free_mem_gb=$(( (total_mem - used_mem) / 1024 ))
    echo "Allocated GPU check: query_gpu=${query_gpu} nvidia_gpu=${gpu_id} total=${total_mem}MiB used=${used_mem}MiB free=${free_mem_gb}GiB required=${REQUIRED_FREE_GB}GiB"
    if (( free_mem_gb >= REQUIRED_FREE_GB )); then
      break
    fi
    if (( GPU_WAIT_TIMEOUT_SECONDS > 0 && SECONDS - gpu_wait_start >= GPU_WAIT_TIMEOUT_SECONDS )); then
      echo "Timed out waiting for the allocated GPU after ${GPU_WAIT_TIMEOUT_SECONDS}s."
      exit 2
    fi
    echo "Allocated GPU is busy outside Slurm accounting; retaining the allocation and retrying in ${GPU_WAIT_INTERVAL_SECONDS}s."
    sleep "${GPU_WAIT_INTERVAL_SECONDS}"
  done
fi

if [[ -f "${TRAIN_SUCCESS_MARKER}" ]]; then
  echo "Training already completed; using ${TRAIN_SUCCESS_MARKER}"
else
  start_gpu_monitor train
  python train.py \
    --model_path="${BASE_MODEL_PATH}" \
    --data_path="${DATASET_PATH}" \
    --method="${METHOD}" \
    --lora_rank="${RANK}" \
    --lora_dropout="${LORA_DROPOUT}" \
    --target_modules \
    q_proj \
    k_proj \
    v_proj \
    o_proj \
    gate_proj \
    up_proj \
    down_proj \
    --max_length="${MAX_LENGTH}" \
    --batch_size="${TRAIN_BATCH_SIZE}" \
    --gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS}" \
    --dataloader_num_workers="${DATALOADER_NUM_WORKERS}" \
    --max_train_samples="${MAX_TRAIN_SAMPLES}" \
    --max_eval_samples="${MAX_EVAL_SAMPLES}" \
    --num_train_epochs="${NUM_TRAIN_EPOCHS}" \
    --learning_rate="${LORA_LR}" \
    --module_learning_rate="${MODULE_LR}" \
    --lr_scheduler_type=constant_with_warmup \
    --warmup_steps="${WARMUP_STEPS}" \
    --weight_decay=0.0 \
    --selection_interval="${SELECTION_INTERVAL}" \
    --compensation_ratio="${COMP_RATIO}" \
    --alpha_score="${ALPHA_SCORE}" \
    --alpha_candidate_ratio="${ALPHA_CANDIDATE_RATIO}" \
    --alpha_sampling_temperature="${ALPHA_SAMPLING_TEMPERATURE}" \
    --alpha_uniform_mix="${ALPHA_UNIFORM_MIX}" \
    --alpha_score_gamma="${ALPHA_SCORE_GAMMA}" \
    --alpha_group_norm="${ALPHA_GROUP_NORM}" \
    --lora_optimizer_reset_strategy="${LORA_OPTIMIZER_RESET_STRATEGY}" \
    --module_optimizer_state_strategy="${MODULE_OPTIMIZER_STATE_STRATEGY}" \
    --lora_optimizer_dtype="${LORA_OPTIMIZER_DTYPE}" \
    --module_optimizer_dtype="${MODULE_OPTIMIZER_DTYPE}" \
    --module_gradient_mode="${MODULE_GRADIENT_MODE}" \
    --residual_rtol="${RESIDUAL_RTOL}" \
    --seed="${SEED}" \
    --gradient_checkpointing="${GRADIENT_CHECKPOINTING}" \
    --save_merged_model="${SAVE_MERGED_MODEL}" \
    --output_dir="${OUTPUT_ROOT}" \
    --run_name="${RUN_NAME}"
  stop_gpu_monitor
  mkdir -p "${MODEL_DIR}"
  date -Iseconds > "${TRAIN_SUCCESS_MARKER}"
fi

if [[ "${EVAL_AFTER_TRAIN}" == "1" ]]; then
  start_gpu_monitor test
  python test_code10.py \
    --model_path="${MODEL_DIR}" \
    --data_path="${EVAL_DATA_PATH}" \
    --max_new_tokens="${TEST_MAX_NEW_TOKENS}" \
    --batch_size="${TEST_BATCH_SIZE}" \
    --num_return_sequences="${TEST_NUM_RETURN_SEQUENCES}"
  stop_gpu_monitor

  python evaluate_code.py \
    --predict_file "${MODEL_DIR}/predictions/humaneval_responses.jsonl"
fi
