#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/mnt/petrelfs/caojie1/projects/loraplusMSeq}
LOG_DIR=${LOG_DIR:-/mnt/dhwfile/raise/user/caojie/loraplusMSeq/logs/codealpaca20k-gemma-3-4b-pt-humaneval}
PARTITION=${1:-belt_road}
CPUS_PER_TASK=${CPUS_PER_TASK:-12}
mkdir -p "${LOG_DIR}"

submission_log="${LOG_DIR}/submitted_codealpaca20k_gemma3_4b_pt_$(date +%Y%m%d_%H%M%S).tsv"

submit_wrap() {
  local label=$1
  local command=$2
  local job_name="${label:0:58}-${PARTITION}"
  local job_id
  job_id=$(sbatch --parsable \
    -p "${PARTITION}" \
    --quotatype=spot \
    --requeue \
    --job-name="${job_name}" \
    --ntasks-per-node=1 \
    --cpus-per-task="${CPUS_PER_TASK}" \
    --gres=gpu:1 \
    --open-mode=append \
    -o "${LOG_DIR}/${label}_${PARTITION}_%j.log" \
    --wrap "cd '${PROJECT_DIR}' && ${command}")
  printf '%s\t%s\t%s\n' "${PARTITION}" "${label}" "${job_id}" | tee -a "${submission_log}"
}

submit_script() {
  local script=$1
  [[ -f "${PROJECT_DIR}/${script}" ]] || { echo "Missing script: ${PROJECT_DIR}/${script}" >&2; exit 1; }
  local label
  label=$(basename "${script}" .sh)
  submit_wrap "${label}" "bash '${script}'"
}

submit_wrap "eval_basemodel_codealpaca_humaneval_gemma3_4b_pt" "MODEL_KEY=gemma3_4b_pt TEST_BATCH_SIZE=\${TEST_BATCH_SIZE:-48} bash eval_basemodel_codealpaca_humaneval.sh"
submit_wrap "eval_basemodel_codealpaca_humaneval_gemma3_4b_it" "MODEL_KEY=gemma3_4b_it TEST_BATCH_SIZE=\${TEST_BATCH_SIZE:-48} bash eval_basemodel_codealpaca_humaneval.sh"
submit_script "exps/codealpaca20k_gemma3/finetune_gemma3_4b_pt_lora_rank32_epoch2_codealpaca20k.sh"
submit_script "exps/codealpaca20k_gemma3/finetune_gemma3_4b_pt_seq_alpha_effective_update_pressure_rank32_ratio0.01386_candidate0.04158_block50_uniform0.1_epoch1_codealpaca20k.sh"
