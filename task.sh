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

METHOD=${METHOD:-alpha}
BASE_MODEL_PATH=${BASE_MODEL_PATH:-meta-llama/Llama-3.1-8B}
DATASET_PATH=${DATASET_PATH:-/mnt/petrelfs/caojie1/projects/CoMoL/datasets/commonsense170k}
EVAL_DATA_PATH=${EVAL_DATA_PATH:-/mnt/petrelfs/caojie1/projects/CoMoL/datasets/math_commonsense}
OUTPUT_ROOT=${OUTPUT_ROOT:-/mnt/dhwfile/raise/user/caojie/loraplusMSeq/outputs/commonsense170k}
RANK=${RANK:-32}
LORA_LR=${LORA_LR:-1e-4}
MODULE_LR=${MODULE_LR:-1e-5}
COMP_RATIO=${COMP_RATIO:-0.005}
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
COMP_DESC=ratio${COMP_RATIO}
if [[ "${METHOD}" == "alpha" && "${ALPHA_CANDIDATE_RATIO}" != "0" ]]; then
  COMP_DESC="${COMP_DESC}-candidate${ALPHA_CANDIDATE_RATIO}"
fi
RUN_NAME=${RUN_NAME:-llama-3-1-8b-seq-${METHOD}-qkvogateupdown-rank${RANK}-commonsense170k-epoch${NUM_TRAIN_EPOCHS}-${COMP_DESC}-block${SELECTION_INTERVAL}-loralr${LORA_LR}-modulelr${MODULE_LR}}
EVAL_AFTER_TRAIN=${EVAL_AFTER_TRAIN:-1}

python train.py \
  --model_path="${BASE_MODEL_PATH}" \
  --data_path="${DATASET_PATH}" \
  --method="${METHOD}" \
  --lora_rank="${RANK}" \
  --target_modules \
  q_proj \
  k_proj \
  v_proj \
  o_proj \
  gate_proj \
  up_proj \
  down_proj \
  --max_length=256 \
  --batch_size=16 \
  --gradient_accumulation_steps=2 \
  --num_train_epochs="${NUM_TRAIN_EPOCHS}" \
  --learning_rate="${LORA_LR}" \
  --module_learning_rate="${MODULE_LR}" \
  --lr_scheduler_type=constant_with_warmup \
  --warmup_steps=200 \
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
  --save_merged_model="${SAVE_MERGED_MODEL}" \
  --output_dir="${OUTPUT_ROOT}" \
  --run_name="${RUN_NAME}"

if [[ "${EVAL_AFTER_TRAIN}" == "1" ]]; then
  MODEL_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
  python test_commonsense.py \
    --model_path="${MODEL_DIR}" \
    --data_path="${EVAL_DATA_PATH}" \
    --max_new_tokens=64 \
    --batch_size=256

  python evaluate_commonsense.py --predict_file "${MODEL_DIR}/predictions/boolq_responses.jsonl"
fi
