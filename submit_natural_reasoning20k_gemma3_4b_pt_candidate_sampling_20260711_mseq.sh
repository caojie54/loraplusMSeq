#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=/mnt/petrelfs/caojie1/projects/loraplusMSeq
LOG_DIR=${PROJECT_DIR}/logs
mkdir -p "${LOG_DIR}"

SCRIPTS=(
  exps/natural_reasoning_20k_gemma3/finetune_gemma3_4b_pt_seq_alpha_effective_update_pressure_rank64_ratio0.02772_candidate0.05544_block50_seed0.sh
  exps/natural_reasoning_20k_gemma3/finetune_gemma3_4b_pt_seq_alpha_effective_update_pressure_rank64_ratio0.02772_candidate0.05544_block50_seed1.sh
  exps/natural_reasoning_20k_gemma3/finetune_gemma3_4b_pt_seq_alpha_effective_update_pressure_rank64_ratio0.02772_candidate0.05544_block50_seed2.sh
  exps/natural_reasoning_20k_gemma3/finetune_gemma3_4b_pt_seq_alpha_effective_update_pressure_rank64_ratio0.02772_candidate0.08316_block50_groupnormtype_seed0.sh
  exps/natural_reasoning_20k_gemma3/finetune_gemma3_4b_pt_seq_alpha_effective_update_pressure_rank64_ratio0.02772_candidate0.08316_block50_groupnormtype_seed1.sh
  exps/natural_reasoning_20k_gemma3/finetune_gemma3_4b_pt_seq_alpha_effective_update_pressure_rank64_ratio0.02772_candidate0.08316_block50_groupnormtype_seed2.sh
  exps/natural_reasoning_20k_gemma3/finetune_gemma3_4b_pt_seq_alpha_effective_update_pressure_rank64_ratio0.02772_candidate0.08316_block50_groupnormglobal_seed0.sh
  exps/natural_reasoning_20k_gemma3/finetune_gemma3_4b_pt_seq_alpha_effective_update_pressure_rank64_ratio0.02772_candidate0.08316_block50_groupnormglobal_seed1.sh
  exps/natural_reasoning_20k_gemma3/finetune_gemma3_4b_pt_seq_alpha_effective_update_pressure_rank64_ratio0.02772_candidate0.08316_block50_groupnormglobal_seed2.sh
  exps/natural_reasoning_20k_gemma3/finetune_gemma3_4b_pt_seq_alpha_effective_update_pressure_rank64_ratio0.02772_candidate0.08316_block50_uniform0.2_seed0.sh
  exps/natural_reasoning_20k_gemma3/finetune_gemma3_4b_pt_seq_alpha_effective_update_pressure_rank64_ratio0.02772_candidate0.08316_block50_uniform0.2_seed1.sh
  exps/natural_reasoning_20k_gemma3/finetune_gemma3_4b_pt_seq_alpha_effective_update_pressure_rank64_ratio0.02772_candidate0.08316_block50_uniform0.2_seed2.sh
  exps/natural_reasoning_20k_gemma3/finetune_gemma3_4b_pt_seq_alpha_gradnorm_pressure_rank64_ratio0.02772_candidate0.08316_block50_seed0.sh
  exps/natural_reasoning_20k_gemma3/finetune_gemma3_4b_pt_seq_alpha_gradnorm_pressure_rank64_ratio0.02772_candidate0.08316_block50_seed1.sh
  exps/natural_reasoning_20k_gemma3/finetune_gemma3_4b_pt_seq_alpha_gradnorm_pressure_rank64_ratio0.02772_candidate0.08316_block50_seed2.sh
)

for script in "${SCRIPTS[@]}"; do
  base=$(basename "${script}" .sh)
  job_name="${base:0:48}"
  sbatch -p raise     --quotatype=spot     --job-name="${job_name}"     --ntasks-per-node=1     --cpus-per-task=24     --gres=gpu:1     -o "${LOG_DIR}/${base}_%j.log"     --wrap "cd ${PROJECT_DIR} && bash ${script}"
done
