#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/mnt/petrelfs/caojie1/projects/loraplusMSeq}
LOG_DIR=${LOG_DIR:-/mnt/dhwfile/raise/user/caojie/loraplusMSeq/logs/basemodel-medmcqa-synthetic-text-to-sql}
PARTITION=${1:-belt_road}
CPUS_PER_TASK=${CPUS_PER_TASK:-12}
mkdir -p "${LOG_DIR}"

models=(
  llama31_8b
  llama31_8b_instruct
  gemma3_1b_pt
  gemma3_1b_it
  gemma3_4b_pt
  gemma3_4b_it
)

tasks=(
  medmcqa
  synthetic_text_to_sql
)

submission_log="${LOG_DIR}/submitted_basemodel_eval_$(date +%Y%m%d_%H%M%S).tsv"

for task in "${tasks[@]}"; do
  for model_key in "${models[@]}"; do
    job_base="mseq-${task}-${model_key}"
    job_name="${job_base:0:58}-${PARTITION}"
    log_base="${task}_${model_key}"
    job_id=$(sbatch --parsable \
      -p "${PARTITION}" \
      --quotatype=spot \
      --requeue \
      --job-name="${job_name}" \
      --ntasks-per-node=1 \
      --cpus-per-task="${CPUS_PER_TASK}" \
      --gres=gpu:1 \
      --open-mode=append \
      -o "${LOG_DIR}/${log_base}_${PARTITION}_%j.log" \
      --wrap "cd '${PROJECT_DIR}' && TASK='${task}' MODEL_KEY='${model_key}' bash eval_basemodel_medmcqa_synthetic_text_to_sql.sh")
    printf '%s\t%s\t%s\t%s\n' "${PARTITION}" "${task}" "${model_key}" "${job_id}" | tee -a "${submission_log}"
  done
done
