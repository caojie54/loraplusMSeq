#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=/mnt/petrelfs/caojie1/projects/loraplusMSeq
LOG_DIR=${PROJECT_DIR}/logs
mkdir -p "${LOG_DIR}"

SCRIPTS=(
  exps/natural_reasoning_20k/finetune_llama_lora_rank64_epoch2_natural_reasoning_20k_loraoptbf16_seed1.sh
  exps/natural_reasoning_20k/finetune_llama_lora_rank64_epoch2_natural_reasoning_20k_loraoptbf16_seed2.sh
  exps/natural_reasoning_20k/finetune_llama_seq_alpha_gradnorm_rank64_ratio0.02_block50_loraoptbf16_moduleoptfp32_reset_offload_seed1.sh
  exps/natural_reasoning_20k/finetune_llama_seq_alpha_gradnorm_rank64_ratio0.02_block50_loraoptbf16_moduleoptfp32_reset_offload_seed2.sh
  exps/natural_reasoning_20k/finetune_llama_seq_alpha_gradnorm_rank64_ratio0.03_block50_loraoptbf16_moduleoptfp32_reset_offload.sh
  exps/natural_reasoning_20k/finetune_llama_seq_alpha_gradnorm_rank64_ratio0.04_block50_loraoptbf16_moduleoptfp32_reset_offload.sh
  exps/commonsense170k/finetune_llama_seq_alpha_gradnorm_pressure_rank64_ratio0.02_block50_loraoptbf16_moduleoptfp32.sh
  exps/commonsense170k/finetune_llama_seq_dynamic_random_rank64_ratio0.02_block50_loraoptbf16_moduleoptfp32.sh
)

for script in "${SCRIPTS[@]}"; do
  base=$(basename "${script}" .sh)
  job_name="${base:0:48}"
  sbatch -p sciverse_agent \
    --job-name="${job_name}" \
    --ntasks-per-node=1 \
    --cpus-per-task=24 \
    --gres=gpu:1 \
    -o "${LOG_DIR}/${base}_%j.log" \
    --wrap "cd ${PROJECT_DIR} && bash ${script}"
done
