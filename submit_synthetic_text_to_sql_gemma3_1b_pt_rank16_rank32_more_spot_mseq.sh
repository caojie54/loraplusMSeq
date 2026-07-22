#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/mnt/petrelfs/caojie1/projects/loraplusMSeq}
LOG_DIR=${LOG_DIR:-/mnt/dhwfile/raise/user/caojie/loraplusMSeq/logs/synthetic-text-to-sql-gemma-3-1b-pt-rank16-rank32-more}
PARTITION=${1:-belt_road}
CPUS_PER_TASK=${CPUS_PER_TASK:-12}
mkdir -p "${LOG_DIR}"

scripts=(
  exps/synthetic_text_to_sql_gemma3/finetune_gemma3_1b_pt_seq_alpha_effective_update_pressure_rank16_ratio0.0130475_candidate0.0391425_block100_uniform0.1_epoch1_synthetic_text_to_sql_full.sh
  exps/synthetic_text_to_sql_gemma3/finetune_gemma3_1b_pt_seq_alpha_effective_update_pressure_rank16_ratio0.0130475_candidate0.0391425_block200_uniform0.1_epoch1_synthetic_text_to_sql_full.sh
  exps/synthetic_text_to_sql_gemma3/finetune_gemma3_1b_pt_seq_alpha_effective_update_pressure_rank16_ratio0.0130475_candidate0.0391425_block300_uniform0.1_epoch1_synthetic_text_to_sql_full.sh
  exps/synthetic_text_to_sql_gemma3/finetune_gemma3_1b_pt_seq_alpha_effective_update_pressure_rank16_ratio0.0130475_candidate0.0391425_block400_uniform0.1_epoch1_synthetic_text_to_sql_full.sh
  exps/synthetic_text_to_sql_gemma3/finetune_gemma3_1b_pt_seq_alpha_effective_update_pressure_rank16_ratio0.0130475_candidate0.0391425_block500_uniform0.1_epoch1_synthetic_text_to_sql_full.sh
  exps/synthetic_text_to_sql_gemma3/finetune_gemma3_1b_pt_lora_rank16_epoch2_synthetic_text_to_sql_full.sh
  exps/synthetic_text_to_sql_gemma3/finetune_gemma3_1b_pt_seq_alpha_effective_update_pressure_rank32_ratio0.026095_candidate0.078285_block600_uniform0.1_epoch1_synthetic_text_to_sql_full.sh
  exps/synthetic_text_to_sql_gemma3/finetune_gemma3_1b_pt_seq_alpha_effective_update_pressure_rank32_ratio0.026095_candidate0.078285_block700_uniform0.1_epoch1_synthetic_text_to_sql_full.sh
  exps/synthetic_text_to_sql_gemma3/finetune_gemma3_1b_pt_seq_alpha_effective_update_pressure_rank32_ratio0.026095_candidate0.078285_block800_uniform0.1_epoch1_synthetic_text_to_sql_full.sh
)

submission_log="${LOG_DIR}/submitted_rank16_rank32_more_$(date +%Y%m%d_%H%M%S).tsv"

for script in "${scripts[@]}"; do
  [[ -f "${PROJECT_DIR}/${script}" ]] || { echo "Missing script: ${PROJECT_DIR}/${script}" >&2; exit 1; }
  base=$(basename "${script}" .sh)
  job_name="${base:0:58}-${PARTITION}"
  job_id=$(sbatch --parsable \
    -p "${PARTITION}" \
    --quotatype=spot \
    --requeue \
    --job-name="${job_name}" \
    --ntasks-per-node=1 \
    --cpus-per-task="${CPUS_PER_TASK}" \
    --gres=gpu:1 \
    --open-mode=append \
    -o "${LOG_DIR}/${base}_${PARTITION}_%j.log" \
    --wrap "cd '${PROJECT_DIR}' && bash '${script}'")
  printf '%s\t%s\t%s\n' "${PARTITION}" "${script}" "${job_id}" | tee -a "${submission_log}"
done
