#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}

BASE_MODEL_PATH=${BASE_MODEL_PATH:-google/gemma-3-4b-pt}
# Derive the default model label from a Hub ID or local model directory.
MODEL_TAG=${BASE_MODEL_PATH%/}
MODEL_TAG=${MODEL_TAG##*/}
MODEL_TAG=${MODEL_TAG,,}
DATASET_PATH=${DATASET_PATH:-/data/user/cj/datasets/MedReason/ours_quality_33000.jsonl}
OUTPUT_ROOT=${OUTPUT_ROOT:-/data/user/cj/experiments/loraplusMSeq/outputs/medreason/${MODEL_TAG}}
METHOD=${METHOD:-lora}
RANK=${RANK:-64}
case "${METHOD}" in
  lora) NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-2} ;;
  static_random|dynamic_random|alpha) NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-1} ;;
  *) echo "Unsupported METHOD=${METHOD}" >&2; exit 2 ;;
esac
SEED=${SEED:-0}
LORA_LR=${LORA_LR:-1e-4}
LORA_OPTIMIZER_DTYPE=${LORA_OPTIMIZER_DTYPE:-bf16}
MODULE_LR=${MODULE_LR:-1e-5}
COMP_RATIO=${COMP_RATIO:-0.02772}
SELECTION_INTERVAL=${SELECTION_INTERVAL:-50}
ALPHA_SCORE=${ALPHA_SCORE:-lora_grad_norm}
ALPHA_CANDIDATE_RATIO=${ALPHA_CANDIDATE_RATIO:-0}
ALPHA_SAMPLING_TEMPERATURE=${ALPHA_SAMPLING_TEMPERATURE:-1.0}
ALPHA_UNIFORM_MIX=${ALPHA_UNIFORM_MIX:-0.1}
ALPHA_SCORE_GAMMA=${ALPHA_SCORE_GAMMA:-1.0}
ALPHA_GROUP_NORM=${ALPHA_GROUP_NORM:-none}
LORA_OPTIMIZER_RESET_STRATEGY=${LORA_OPTIMIZER_RESET_STRATEGY:-keep}
MODULE_OPTIMIZER_STATE_STRATEGY=${MODULE_OPTIMIZER_STATE_STRATEGY:-reset_offload}
MODULE_OPTIMIZER_DTYPE=${MODULE_OPTIMIZER_DTYPE:-fp32}
MODULE_GRADIENT_MODE=${MODULE_GRADIENT_MODE:-full}
RESIDUAL_RTOL=${RESIDUAL_RTOL:-1e-4}

case "${ALPHA_SCORE}" in
  lora_grad_norm|lora_grad_norm_min|lora_update_ratio|lora_effective_update_pressure) ;;
  *) echo "Unsupported ALPHA_SCORE=${ALPHA_SCORE}" >&2; exit 2 ;;
esac

METHOD_DESC=${METHOD}
if [[ "${METHOD}" != lora ]]; then
  METHOD_DESC="seq-${METHOD}"
  if [[ "${METHOD}" == alpha ]]; then
    METHOD_DESC="${METHOD_DESC}-${ALPHA_SCORE}-candidate${ALPHA_CANDIDATE_RATIO}"
  fi
  METHOD_DESC="${METHOD_DESC}-ratio${COMP_RATIO}-block${SELECTION_INTERVAL}"
fi
DEFAULT_RUN_NAME="${MODEL_TAG}-${METHOD_DESC}-qkvogateupdown-rank${RANK}-medreason-epoch${NUM_TRAIN_EPOCHS}-seed${SEED}-loraopt${LORA_OPTIMIZER_DTYPE}-loralr${LORA_LR}"
if [[ "${METHOD}" != lora ]]; then
  DEFAULT_RUN_NAME="${DEFAULT_RUN_NAME}-moduleopt${MODULE_OPTIMIZER_DTYPE}-${MODULE_OPTIMIZER_STATE_STRATEGY}-modulelr${MODULE_LR}"
fi
RUN_NAME=${RUN_NAME:-${DEFAULT_RUN_NAME}}
source scripts/select_free_gpu.sh

python train.py \
  --data_format medreason \
  --model_path "${BASE_MODEL_PATH}" \
  --data_path "${DATASET_PATH}" \
  --method "${METHOD}" \
  --lora_rank "${RANK}" \
  --lora_dropout "${LORA_DROPOUT:-0.0}" \
  --target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
  --max_length "${MAX_LENGTH:-1536}" \
  --batch_size "${TRAIN_BATCH_SIZE:-4}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-8}" \
  --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
  --learning_rate "${LORA_LR}" \
  --lora_optimizer_dtype "${LORA_OPTIMIZER_DTYPE}" \
  --module_learning_rate "${MODULE_LR}" \
  --compensation_ratio "${COMP_RATIO}" \
  --selection_interval "${SELECTION_INTERVAL}" \
  --alpha_score "${ALPHA_SCORE}" \
  --alpha_candidate_ratio "${ALPHA_CANDIDATE_RATIO}" \
  --alpha_sampling_temperature "${ALPHA_SAMPLING_TEMPERATURE}" \
  --alpha_uniform_mix "${ALPHA_UNIFORM_MIX}" \
  --alpha_score_gamma "${ALPHA_SCORE_GAMMA}" \
  --alpha_group_norm "${ALPHA_GROUP_NORM}" \
  --lora_optimizer_reset_strategy "${LORA_OPTIMIZER_RESET_STRATEGY}" \
  --module_optimizer_state_strategy "${MODULE_OPTIMIZER_STATE_STRATEGY}" \
  --module_optimizer_dtype "${MODULE_OPTIMIZER_DTYPE}" \
  --module_gradient_mode "${MODULE_GRADIENT_MODE}" \
  --residual_rtol "${RESIDUAL_RTOL}" \
  --lr_scheduler_type constant_with_warmup \
  --warmup_steps "${WARMUP_STEPS:-50}" \
  --weight_decay 0.0 \
  --seed "${SEED}" \
  --gradient_checkpointing "${GRADIENT_CHECKPOINTING:-false}" \
  --save_merged_model "${SAVE_MERGED_MODEL:-true}" \
  --max_train_samples "${MAX_TRAIN_SAMPLES:-0}" \
  --output_dir "${OUTPUT_ROOT}" \
  --run_name "${RUN_NAME}"

echo "Training complete: ${OUTPUT_ROOT}/${RUN_NAME}"
echo "For MedReason evaluation, serve ${OUTPUT_ROOT}/${RUN_NAME}/merged with SGLang."
echo "Then run MODEL_DIR=${OUTPUT_ROOT}/${RUN_NAME}/merged bash eval_med_shell/eval_medreason.sh"
