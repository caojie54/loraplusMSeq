#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/mnt/petrelfs/caojie1/projects/loraplusMSeq}
LOG_DIR=${LOG_DIR:-/mnt/dhwfile/raise/user/caojie/loraplusMSeq/logs/medmcqa-and-synthetic-text-to-sql-full-outputonly}
OUTPUT_BASE=${OUTPUT_BASE:-/mnt/dhwfile/raise/user/caojie/loraplusMSeq/outputs/synthetic-text-to-sql-full-outputonly}
PARTITION=${1:-raise}
EXPERIMENT=${2:-lora}
EXCLUDE_NODES=${EXCLUDE_NODES:-}
INCLUDE_NODES=${INCLUDE_NODES:-}
CPUS_PER_TASK=${CPUS_PER_TASK:-12}

partitions=(sciverse_agent raise belt_road)
if [[ ! " ${partitions[*]} " =~ " ${PARTITION} " ]]; then
  printf 'Unknown partition: %s\n' "${PARTITION}" >&2
  exit 2
fi

case "${EXPERIMENT}" in
  lora)
    script="exps/synthetic_text_to_sql/finetune_llama_lora_rank32_epoch2_synthetic_text_to_sql_full.sh"
    job_tag=lora
    epoch_tag=e2
    ;;
  lora_effective_update_pressure|effective_update_pressure)
    EXPERIMENT=lora_effective_update_pressure
    script="exps/synthetic_text_to_sql/finetune_llama_seq_alpha_effective_update_pressure_rank32_ratio0.01_candidate0.03_block50_uniform0.1_epoch1_synthetic_text_to_sql_full.sh"
    job_tag=eup
    epoch_tag=e1
    ;;
  dynamic_random)
    script="exps/synthetic_text_to_sql/finetune_llama_seq_dynamic_random_rank32_ratio0.01_block50_epoch1_synthetic_text_to_sql_full.sh"
    job_tag=dynamic
    epoch_tag=e1
    ;;
  *)
    printf 'Unknown experiment: %s\n' "${EXPERIMENT}" >&2
    exit 2
    ;;
esac
if [[ ! -f "${PROJECT_DIR}/${script}" ]]; then
  printf 'Missing script: %s\n' "${PROJECT_DIR}/${script}" >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"
submission_log="${LOG_DIR}/submitted_synthetic_text_to_sql_${EXPERIMENT}_$(date +%Y%m%d_%H%M%S).tsv"
printf 'partition\tdataset\tmethod\tjob_id\toutput_root\n' > "${submission_log}"

output_root="${OUTPUT_BASE}"
sbatch_args=(
  --parsable
  -p "${PARTITION}"
  --quotatype=spot
  --requeue
  --job-name="mseq-sql-${job_tag}-r32-${epoch_tag}-${PARTITION}"
  --ntasks-per-node=1
  --cpus-per-task="${CPUS_PER_TASK}"
  --gres=gpu:1
  --open-mode=append
  -o "${LOG_DIR}/synthetic_text_to_sql_${EXPERIMENT}_${PARTITION}_%j.log"
  --export="ALL,OUTPUT_ROOT=${output_root}"
  --wrap="cd '${PROJECT_DIR}' && bash '${script}'"
)
if [[ -n "${EXCLUDE_NODES}" ]]; then
  sbatch_args+=(--exclude="${EXCLUDE_NODES}")
fi
if [[ -n "${INCLUDE_NODES}" ]]; then
  sbatch_args+=(--nodelist="${INCLUDE_NODES}")
fi
job_id=$(sbatch "${sbatch_args[@]}")
printf '%s\t%s\t%s\t%s\t%s\n' "${PARTITION}" synthetic_text_to_sql "${EXPERIMENT}" "${job_id}" "${output_root}" | tee -a "${submission_log}"
printf 'Submission manifest: %s\n' "${submission_log}"
