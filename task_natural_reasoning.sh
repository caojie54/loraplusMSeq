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

METHOD=${METHOD:-alpha}
BASE_MODEL_PATH=${BASE_MODEL_PATH:-meta-llama/Llama-3.1-8B}
DATASET_PATH=${DATASET_PATH:-/mnt/petrelfs/caojie1/projects/CoMoL/datasets/natural_reasoning_20k}
EVAL_DATA_PATH=${EVAL_DATA_PATH:-/mnt/petrelfs/caojie1/projects/CoMoL/datasets/natural_reasoning_eval}
OUTPUT_ROOT=${OUTPUT_ROOT:-/mnt/dhwfile/raise/user/caojie/loraplusMSeq/outputs/natural-reasoning-20k}
RANK=${RANK:-64}
LORA_LR=${LORA_LR:-1e-4}
MODULE_LR=${MODULE_LR:-1e-5}
LORA_DROPOUT=${LORA_DROPOUT:-0.0}
SEED=${SEED:-0}
COMP_RATIO=${COMP_RATIO:-0.02}
SELECTION_INTERVAL=${SELECTION_INTERVAL:-625}
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
  NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-1}
  SAVE_MERGED_MODEL=${SAVE_MERGED_MODEL:-false}
else
  NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-1}
  SAVE_MERGED_MODEL=${SAVE_MERGED_MODEL:-true}
fi

METHOD_DESC="${METHOD}"
if [[ "${METHOD}" == "alpha" ]]; then
  if [[ "${ALPHA_SCORE}" == "lora_grad_norm" ]]; then
    METHOD_DESC="${METHOD_DESC}-gradnorm"
  elif [[ "${ALPHA_SCORE}" == "lora_grad_norm_min" ]]; then
    METHOD_DESC="${METHOD_DESC}-gradnormmin"
  elif [[ "${ALPHA_SCORE}" == "lora_effective_update_pressure" ]]; then
    METHOD_DESC="${METHOD_DESC}-effectivepressure"
  else
    METHOD_DESC="${METHOD_DESC}-${ALPHA_SCORE}"
  fi
  if [[ "${ALPHA_CANDIDATE_RATIO}" != "0" ]]; then
    METHOD_DESC="${METHOD_DESC}-candidate${ALPHA_CANDIDATE_RATIO}"
  fi
fi
if [[ "${MODULE_OPTIMIZER_STATE_STRATEGY}" != "reset_offload" ]]; then
  METHOD_DESC="${METHOD_DESC}-moduleopt-${MODULE_OPTIMIZER_STATE_STRATEGY}"
fi
if [[ "${LORA_OPTIMIZER_DTYPE}" != "bf16" ]]; then
  METHOD_DESC="${METHOD_DESC}-loraopt-${LORA_OPTIMIZER_DTYPE}"
fi
if [[ "${MODULE_OPTIMIZER_DTYPE}" != "bf16" ]]; then
  METHOD_DESC="${METHOD_DESC}-moduleoptdtype-${MODULE_OPTIMIZER_DTYPE}"
fi
if [[ "${MODULE_GRADIENT_MODE}" != "full" ]]; then
  METHOD_DESC="${METHOD_DESC}-modulegrad-${MODULE_GRADIENT_MODE}"
fi

RUN_NAME=${RUN_NAME:-llama-3-1-8b-seq-${METHOD_DESC}-qkvogateupdown-rank${RANK}-natural-reasoning-20k-epoch${NUM_TRAIN_EPOCHS}-ratio${COMP_RATIO}-block${SELECTION_INTERVAL}-loralr${LORA_LR}-modulelr${MODULE_LR}}
EVAL_AFTER_TRAIN=${EVAL_AFTER_TRAIN:-1}
MAX_LENGTH=${MAX_LENGTH:-1536}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-4}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-8}
WARMUP_STEPS=${WARMUP_STEPS:-50}
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-false}
REQUIRED_FREE_GB=${REQUIRED_FREE_GB:-75}
TEST_MAX_NEW_TOKENS=${TEST_MAX_NEW_TOKENS:-1536}
TEST_BATCH_SIZE=${TEST_BATCH_SIZE:-64}
NATURAL_REASONING_BENCHMARKS=${NATURAL_REASONING_BENCHMARKS:-gpqa_diamond math_500 mmlu_pro_500}

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
  IFS=, read -r gpu_id total_mem used_mem < <(
    nvidia-smi -i "${query_gpu}" --query-gpu=index,memory.total,memory.used --format=csv,noheader,nounits
  )
  gpu_id="${gpu_id//[[:space:]]/}"
  total_mem="${total_mem//[[:space:]]/}"
  used_mem="${used_mem//[[:space:]]/}"
  free_mem_gb=$(( (total_mem - used_mem) / 1024 ))
  echo "Allocated GPU check: query_gpu=${query_gpu} nvidia_gpu=${gpu_id} total=${total_mem}MiB used=${used_mem}MiB free=${free_mem_gb}GiB required=${REQUIRED_FREE_GB}GiB"
  if (( free_mem_gb < REQUIRED_FREE_GB )); then
    echo "Allocated GPU does not have enough free memory; exiting for resubmission."
    exit 2
  fi
fi

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

if [[ "${EVAL_AFTER_TRAIN}" == "1" ]]; then
  MODEL_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
  # shellcheck disable=SC2086
  python test_natural_reasoning.py \
    --model_path="${MODEL_DIR}" \
    --data_path="${EVAL_DATA_PATH}" \
    --benchmarks ${NATURAL_REASONING_BENCHMARKS} \
    --max_new_tokens="${TEST_MAX_NEW_TOKENS}" \
    --batch_size="${TEST_BATCH_SIZE}"

  # shellcheck disable=SC2086
  python evaluate_natural_reasoning.py \
    --prediction_dir "${MODEL_DIR}/predictions" \
    --benchmarks ${NATURAL_REASONING_BENCHMARKS}
fi
