#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

source /mnt/petrelfs/caojie1/anaconda3/etc/profile.d/conda.sh
conda activate comol

export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export HF_HOME=${HF_HOME:-/mnt/dhwfile/raise/user/caojie/huggingface}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}

MODEL_DIR=${MODEL_DIR:-/mnt/dhwfile/raise/user/caojie/loraplusMSeq/outputs/natural-reasoning-20k/llama-3-1-8b-lora-qkvogateupdown-rank64-natural-reasoning-20k-epoch2-loraoptbf16-loralr1e-4}
EVAL_DATA_PATH=${EVAL_DATA_PATH:-/mnt/petrelfs/caojie1/projects/CoMoL/datasets/natural_reasoning_eval}
OUTPUT_DIR=${OUTPUT_DIR:-${MODEL_DIR}/eval_bs64_greedy_default_20260701}
TEST_MAX_NEW_TOKENS=${TEST_MAX_NEW_TOKENS:-1536}
TEST_BATCH_SIZE=${TEST_BATCH_SIZE:-64}
NATURAL_REASONING_BENCHMARKS=${NATURAL_REASONING_BENCHMARKS:-gpqa_diamond math_500 mmlu_pro_500}
REQUIRED_FREE_GB=${REQUIRED_FREE_GB:-65}

echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_NODELIST=${SLURM_NODELIST:-}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "MODEL_DIR=${MODEL_DIR}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "Inference: default greedy search from test_natural_reasoning.py; no --do_sample argument is passed."
nvidia-smi || true

if [[ "${REQUIRED_FREE_GB}" != "0" ]]; then
  allocated_gpu="${CUDA_VISIBLE_DEVICES:-${SLURM_JOB_GPUS:-0}}"
  allocated_gpu="${allocated_gpu%%,*}"
  allocated_gpu="${allocated_gpu//[[:space:]]/}"
  if [[ ! "${allocated_gpu}" =~ ^[0-9]+$ ]]; then allocated_gpu=0; fi
  IFS=, read -r gpu_id total_mem used_mem < <(nvidia-smi -i "${allocated_gpu}" --query-gpu=index,memory.total,memory.used --format=csv,noheader,nounits)
  gpu_id="${gpu_id//[[:space:]]/}"; total_mem="${total_mem//[[:space:]]/}"; used_mem="${used_mem//[[:space:]]/}"
  free_mem_gb=$(( (total_mem - used_mem) / 1024 ))
  echo "Allocated GPU check: query_gpu=${allocated_gpu} nvidia_gpu=${gpu_id} total=${total_mem}MiB used=${used_mem}MiB free=${free_mem_gb}GiB required=${REQUIRED_FREE_GB}GiB"
  if (( free_mem_gb < REQUIRED_FREE_GB )); then
    echo "Allocated GPU does not have enough free memory; exiting for resubmission."
    exit 2
  fi
fi

# shellcheck disable=SC2086
python test_natural_reasoning.py \
  --model_path "${MODEL_DIR}" \
  --data_path "${EVAL_DATA_PATH}" \
  --benchmarks ${NATURAL_REASONING_BENCHMARKS} \
  --max_new_tokens "${TEST_MAX_NEW_TOKENS}" \
  --batch_size "${TEST_BATCH_SIZE}" \
  --output_dir "${OUTPUT_DIR}"

# shellcheck disable=SC2086
python evaluate_natural_reasoning.py \
  --prediction_dir "${OUTPUT_DIR}/predictions" \
  --benchmarks ${NATURAL_REASONING_BENCHMARKS}
