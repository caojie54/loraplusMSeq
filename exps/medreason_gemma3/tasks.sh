#!/usr/bin/env bash
set -euo pipefail

# Resolve task paths relative to this file, regardless of the caller's directory.
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
tasks=(
#   finetune_gemma3_4b_pt_seq_alpha_effective_update_pressure_rank64_ratio0.02772_candidate0.08316_block100_seed0.sh
#   finetune_gemma3_4b_pt_seq_alpha_effective_update_pressure_rank64_ratio0.02772_candidate0.08316_block200_seed0.sh
  finetune_gemma3_4b_pt_seq_alpha_effective_update_pressure_rank64_ratio0.02772_candidate0.08316_block300_seed0.sh
  finetune_gemma3_4b_pt_seq_alpha_effective_update_pressure_rank64_ratio0.02772_candidate0.08316_block400_seed0.sh
)

# Check the entire queue before starting any training.
for task in "${tasks[@]}"; do
  if [[ ! -f "${script_dir}/${task}" ]]; then
    echo "Task script not found: ${script_dir}/${task}" >&2
    exit 1
  fi
done

# Select once for the whole queue. Match the current tasks' 70 GiB default.
REQUIRED_FREE_GB=${REQUIRED_FREE_GB:-70}
source "${script_dir}/../../scripts/select_free_gpu.sh"
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" || "${CUDA_VISIBLE_DEVICES}" == *,* || "${CUDA_VISIBLE_DEVICES}" == -1 ]]; then
  echo "The task queue requires one GPU; when REQUIRED_FREE_GB=0, set CUDA_VISIBLE_DEVICES to a single GPU." >&2
  exit 2
fi
readonly task_queue_gpu="${CUDA_VISIBLE_DEVICES}"
echo "All queued tasks will use CUDA_VISIBLE_DEVICES=${task_queue_gpu}"

for task in "${tasks[@]}"; do
  echo "Starting: ${task}"
  # Zero disables the child launcher's GPU discovery/memory check.
  CUDA_VISIBLE_DEVICES="${task_queue_gpu}" REQUIRED_FREE_GB=0 bash "${script_dir}/${task}"
  echo "Finished: ${task}"
done
