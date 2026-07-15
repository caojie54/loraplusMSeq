#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=/mnt/petrelfs/caojie1/projects/loraplusMSeq
LOG_DIR=${PROJECT_DIR}/logs
mkdir -p "${LOG_DIR}"

SCRIPTS=(
  exps/natural_reasoning_20k_gemma4/finetune_gemma4_E2B_seq_alpha_effective_update_pressure_rank64_ratio0.02_block50_seed0.sh
  exps/natural_reasoning_20k_gemma4/finetune_gemma4_E2B_seq_alpha_effective_update_pressure_rank64_ratio0.02_block50_seed1.sh
  exps/natural_reasoning_20k_gemma4/finetune_gemma4_E2B_seq_alpha_effective_update_pressure_rank64_ratio0.02_block50_seed2.sh
  exps/natural_reasoning_20k_gemma4/finetune_gemma4_E2B_seq_alpha_effective_update_pressure_rank64_ratio0.02_candidate0.06_uniform0.05_block50_seed0.sh
  exps/natural_reasoning_20k_gemma4/finetune_gemma4_E2B_seq_alpha_effective_update_pressure_rank64_ratio0.02_candidate0.06_uniform0.05_block50_seed1.sh
  exps/natural_reasoning_20k_gemma4/finetune_gemma4_E2B_seq_alpha_effective_update_pressure_rank64_ratio0.02_candidate0.06_uniform0.05_block50_seed2.sh
  exps/natural_reasoning_20k_gemma4/finetune_gemma4_E2B_seq_alpha_effective_update_pressure_rank64_ratio0.02_candidate0.06_uniform0.1_block50_seed0.sh
  exps/natural_reasoning_20k_gemma4/finetune_gemma4_E2B_seq_alpha_effective_update_pressure_rank64_ratio0.02_candidate0.06_uniform0.1_block50_seed1.sh
  exps/natural_reasoning_20k_gemma4/finetune_gemma4_E2B_seq_alpha_effective_update_pressure_rank64_ratio0.02_candidate0.06_uniform0.1_block50_seed2.sh
  exps/natural_reasoning_20k_gemma4/finetune_gemma4_E2B_seq_alpha_effective_update_pressure_rank64_ratio0.02_candidate0.06_uniform0.2_block50_seed0.sh
  exps/natural_reasoning_20k_gemma4/finetune_gemma4_E2B_seq_alpha_effective_update_pressure_rank64_ratio0.02_candidate0.06_uniform0.2_block50_seed1.sh
  exps/natural_reasoning_20k_gemma4/finetune_gemma4_E2B_seq_alpha_effective_update_pressure_rank64_ratio0.02_candidate0.06_uniform0.2_block50_seed2.sh
)

for script in "${SCRIPTS[@]}"; do
  base=$(basename "${script}" .sh)
  job_name="${base:0:48}"
  sbatch -p belt_road \
    --quotatype=spot \
    --exclude=SH-IDCA1404-10-140-54-15 \
    --job-name="${job_name}" \
    --ntasks-per-node=1 \
    --cpus-per-task=24 \
    --gres=gpu:1 \
    -o "${LOG_DIR}/${base}_%j.log" \
    --wrap "cd ${PROJECT_DIR} && bash ${script}"
done
