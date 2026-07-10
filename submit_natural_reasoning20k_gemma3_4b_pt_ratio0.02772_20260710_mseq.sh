#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=/mnt/petrelfs/caojie1/projects/loraplusMSeq
LOG_DIR=${PROJECT_DIR}/logs
mkdir -p "${LOG_DIR}"

SCRIPTS=(
  exps/natural_reasoning_20k_gemma3/finetune_gemma3_4b_pt_lora_rank64_epoch2_natural_reasoning_20k_seed0.sh
  exps/natural_reasoning_20k_gemma3/finetune_gemma3_4b_pt_lora_rank64_epoch2_natural_reasoning_20k_seed1.sh
  exps/natural_reasoning_20k_gemma3/finetune_gemma3_4b_pt_lora_rank64_epoch2_natural_reasoning_20k_seed2.sh
  exps/natural_reasoning_20k_gemma3/finetune_gemma3_4b_pt_seq_dynamic_random_rank64_ratio0.02772_block50_seed0.sh
  exps/natural_reasoning_20k_gemma3/finetune_gemma3_4b_pt_seq_dynamic_random_rank64_ratio0.02772_block50_seed1.sh
  exps/natural_reasoning_20k_gemma3/finetune_gemma3_4b_pt_seq_dynamic_random_rank64_ratio0.02772_block50_seed2.sh
  exps/natural_reasoning_20k_gemma3/finetune_gemma3_4b_pt_seq_alpha_gradnorm_pressure_rank64_ratio0.02772_block50_seed0.sh
  exps/natural_reasoning_20k_gemma3/finetune_gemma3_4b_pt_seq_alpha_gradnorm_pressure_rank64_ratio0.02772_block50_seed1.sh
  exps/natural_reasoning_20k_gemma3/finetune_gemma3_4b_pt_seq_alpha_gradnorm_pressure_rank64_ratio0.02772_block50_seed2.sh
)

for script in "${SCRIPTS[@]}"; do
  base=$(basename "${script}" .sh)
  job_name="${base:0:48}"
  sbatch -p sciverse_agent \
    --exclude=SH-IDC1-10-140-37-49 \
    --job-name="${job_name}" \
    --ntasks-per-node=1 \
    --cpus-per-task=24 \
    --gres=gpu:1 \
    -o "${LOG_DIR}/${base}_%j.log" \
    --wrap "cd ${PROJECT_DIR} && bash ${script}"
done
