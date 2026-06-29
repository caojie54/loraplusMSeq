#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=/mnt/petrelfs/caojie1/projects/loraplusMSeq
LOG_DIR=${PROJECT_DIR}/logs
EXCLUDE_NODES=${EXCLUDE_NODES:-}

mkdir -p "${LOG_DIR}"

scripts=(
  exps/natural_reasoning_20k/finetune_llama_lora_rank64_epoch2_natural_reasoning_20k_loraoptfp32.sh
  exps/natural_reasoning_20k/finetune_llama_lora_rank64_epoch2_natural_reasoning_20k_loraoptbf16.sh
  exps/natural_reasoning_20k/finetune_llama_seq_dynamic_random_rank64_ratio0.02_block625_loraoptbf16_moduleoptfp32.sh
  exps/natural_reasoning_20k/finetune_llama_seq_alpha_gradnorm_rank64_ratio0.02_block50_loraoptbf16_moduleoptfp32_persistent_offload.sh
  exps/natural_reasoning_20k/finetune_llama_seq_alpha_gradnorm_rank64_ratio0.02_block50_loraoptbf16_moduleoptfp32_reset_offload.sh
  exps/natural_reasoning_20k/finetune_llama_seq_dynamic_random_rank64_ratio0.02_block625_loraoptfp32_moduleoptfp32.sh
)

exclude_arg=()
if [[ -n "${EXCLUDE_NODES}" ]]; then
  exclude_arg=(--exclude="${EXCLUDE_NODES}")
fi

for script in "${scripts[@]}"; do
  base=$(basename "${script}" .sh)
  job_name="nr20k_${base}"
  sbatch \
    -p sciverse_agent \
    --job-name="${job_name:0:48}" \
    --ntasks-per-node=1 \
    --cpus-per-task=24 \
    --gres=gpu:1 \
    ${exclude_arg[@]+"${exclude_arg[@]}"} \
    -o "${LOG_DIR}/natural_reasoning_20k_${base}_%j.log" \
    --wrap "cd ${PROJECT_DIR} && echo SLURM_JOB_ID=\${SLURM_JOB_ID:-} && echo SLURM_NODELIST=\${SLURM_NODELIST:-} && echo CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-} && nvidia-smi && srun --ntasks=1 --cpus-per-task=24 --gres=gpu:1 bash ${script}"
done
