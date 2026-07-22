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

TASK=${TASK:?Set TASK to medmcqa or synthetic_text_to_sql}
MODEL_KEY=${MODEL_KEY:?Set MODEL_KEY}

case "${MODEL_KEY}" in
  llama31_8b)
    BASE_MODEL_PATH=${BASE_MODEL_PATH:-meta-llama/Llama-3.1-8B}
    MODEL_SLUG=llama-3-1-8b
    ;;
  llama31_8b_instruct)
    BASE_MODEL_PATH=${BASE_MODEL_PATH:-meta-llama/Llama-3.1-8B-Instruct}
    MODEL_SLUG=llama-3-1-8b-instruct
    ;;
  gemma3_1b_pt)
    BASE_MODEL_PATH=${BASE_MODEL_PATH:-google/gemma-3-1b-pt}
    MODEL_SLUG=gemma-3-1b-pt
    ;;
  gemma3_1b_it)
    BASE_MODEL_PATH=${BASE_MODEL_PATH:-google/gemma-3-1b-it}
    MODEL_SLUG=gemma-3-1b-it
    ;;
  gemma3_4b_pt)
    BASE_MODEL_PATH=${BASE_MODEL_PATH:-google/gemma-3-4b-pt}
    MODEL_SLUG=gemma-3-4b-pt
    ;;
  gemma3_4b_it)
    BASE_MODEL_PATH=${BASE_MODEL_PATH:-google/gemma-3-4b-it}
    MODEL_SLUG=gemma-3-4b-it
    ;;
  *)
    echo "Unknown MODEL_KEY=${MODEL_KEY}" >&2
    exit 1
    ;;
esac

case "${TASK}" in
  medmcqa)
    EVAL_DATA_PATH=${EVAL_DATA_PATH:-/mnt/petrelfs/caojie1/projects/CoMoL/datasets/medmcqa_eval}
    OUTPUT_ROOT=${OUTPUT_ROOT:-/mnt/dhwfile/raise/user/caojie/loraplusMSeq/outputs/medmcqa-basemodel-eval}
    TEST_MAX_NEW_TOKENS=${TEST_MAX_NEW_TOKENS:-8}
    TEST_BATCH_SIZE=${TEST_BATCH_SIZE:-800}
    OUTPUT_DIR="${OUTPUT_ROOT}/${MODEL_SLUG}"
    python test_medmcqa.py \
      --model_path="${BASE_MODEL_PATH}" \
      --data_path="${EVAL_DATA_PATH}" \
      --max_new_tokens="${TEST_MAX_NEW_TOKENS}" \
      --batch_size="${TEST_BATCH_SIZE}" \
      --output_dir="${OUTPUT_DIR}"
    python evaluate_medmcqa.py \
      --prediction_dir "${OUTPUT_DIR}/predictions"
    ;;
  synthetic_text_to_sql)
    EVAL_DATA_PATH=${EVAL_DATA_PATH:-/mnt/petrelfs/caojie1/projects/CoMoL/datasets/synthetic_text_to_sql_eval}
    OUTPUT_ROOT=${OUTPUT_ROOT:-/mnt/dhwfile/raise/user/caojie/loraplusMSeq/outputs/synthetic-text-to-sql-basemodel-eval}
    TEST_MAX_NEW_TOKENS=${TEST_MAX_NEW_TOKENS:-128}
    if [[ "${MODEL_KEY}" == "gemma3_1b_pt" || "${MODEL_KEY}" == "gemma3_1b_it" ]]; then
      TEST_BATCH_SIZE=${TEST_BATCH_SIZE:-800}
    else
      TEST_BATCH_SIZE=${TEST_BATCH_SIZE:-416}
    fi
    OUTPUT_DIR="${OUTPUT_ROOT}/${MODEL_SLUG}"
    python test_synthetic_text_to_sql.py \
      --model_path="${BASE_MODEL_PATH}" \
      --data_path="${EVAL_DATA_PATH}" \
      --max_new_tokens="${TEST_MAX_NEW_TOKENS}" \
      --batch_size="${TEST_BATCH_SIZE}" \
      --output_dir="${OUTPUT_DIR}"
    python evaluate_synthetic_text_to_sql.py \
      --prediction_dir "${OUTPUT_DIR}/predictions"
    ;;
  *)
    echo "Unknown TASK=${TASK}" >&2
    exit 1
    ;;
esac
