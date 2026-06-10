#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

source /mnt/petrelfs/caojie1/anaconda3/etc/profile.d/conda.sh
conda activate comol

METHOD=${METHOD:-alpha}
BASE_MODEL_PATH=${BASE_MODEL_PATH:-meta-llama/Llama-3.1-8B}
DATASET_PATH=${DATASET_PATH:-/mnt/petrelfs/caojie1/projects/CoMoL/datasets/commonsense170k}
EVAL_DATA_PATH=${EVAL_DATA_PATH:-/mnt/petrelfs/caojie1/projects/CoMoL/datasets/math_commonsense}
OUTPUT_ROOT=${OUTPUT_ROOT:-/mnt/dhwfile/raise/user/caojie/loraplusMSeq/outputs/commonsense170k}
RANK=${RANK:-32}
COMP_TOP_K=${COMP_TOP_K:-8}
SELECTION_INTERVAL=${SELECTION_INTERVAL:-50}
if [[ "${METHOD}" == "lora" ]]; then
  NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-2}
else
  NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-1}
fi
RUN_NAME=${RUN_NAME:-llama-3-1-8b-seq-${METHOD}-qkvogateupdown-rank${RANK}-commonsense170k-epoch${NUM_TRAIN_EPOCHS}}
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
  --learning_rate=1e-4 \
  --lr_scheduler_type=constant_with_warmup \
  --warmup_steps=200 \
  --weight_decay=0.0 \
  --selection_interval="${SELECTION_INTERVAL}" \
  --compensation_top_k="${COMP_TOP_K}" \
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

