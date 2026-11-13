#!/usr/bin/env bash
# Source this helper to export the selected GPU into the calling shell.
REQUIRED_FREE_GB=${REQUIRED_FREE_GB:-75}
if [[ ! "${REQUIRED_FREE_GB}" =~ ^[0-9]+$ ]]; then
  echo "REQUIRED_FREE_GB must be a non-negative integer (GiB)." >&2
  exit 2
fi

if (( 10#${REQUIRED_FREE_GB} > 0 )); then
  required_free_mib=$(( 10#${REQUIRED_FREE_GB} * 1024 ))
  # Honor an explicit visibility mask, including an empty mask (no GPUs).
  if [[ -v CUDA_VISIBLE_DEVICES ]]; then
    candidate_gpus="${CUDA_VISIBLE_DEVICES}"
  elif [[ -n "${SLURM_JOB_GPUS:-}" ]]; then
    candidate_gpus="${SLURM_JOB_GPUS}"
  else
    if ! candidate_gpus=$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits); then
      echo "Unable to discover GPUs with nvidia-smi." >&2
      exit 2
    fi
    candidate_gpus="${candidate_gpus//$'\n'/,}"
  fi

  echo "GPU candidates=${candidate_gpus}; required free memory=${REQUIRED_FREE_GB}GiB"
  IFS=, read -r -a gpu_candidates <<< "${candidate_gpus}"
  selected_gpu=""
  best_free_mib=-1
  for candidate in "${gpu_candidates[@]}"; do
    candidate="${candidate//[[:space:]]/}"
    [[ -n "${candidate}" ]] || continue
    if ! gpu_info=$(nvidia-smi -i "${candidate}" --query-gpu=index,uuid,memory.free --format=csv,noheader,nounits); then
      echo "Unable to query GPU ${candidate}; skipping." >&2
      continue
    fi
    IFS=, read -r gpu_id gpu_uuid free_mib <<< "${gpu_info}"
    gpu_id="${gpu_id//[[:space:]]/}"
    gpu_uuid="${gpu_uuid//[[:space:]]/}"
    free_mib="${free_mib//[[:space:]]/}"
    if [[ ! "${free_mib}" =~ ^[0-9]+$ || "${gpu_uuid}" != GPU-* ]]; then
      echo "Invalid GPU memory/UUID response for ${candidate}; skipping." >&2
      continue
    fi
    free_mib=$(( 10#${free_mib} ))
    echo "GPU ${gpu_id} (${gpu_uuid}): free=${free_mib}MiB"
    if (( free_mib >= required_free_mib && free_mib > best_free_mib )); then
      selected_gpu="${gpu_uuid}"
      best_free_mib=${free_mib}
    fi
  done

  if [[ -z "${selected_gpu}" ]]; then
    echo "No candidate GPU has at least ${REQUIRED_FREE_GB}GiB free; exiting." >&2
    exit 2
  fi
  # UUID avoids differences between CUDA device order and nvidia-smi indices.
  export CUDA_VISIBLE_DEVICES="${selected_gpu}"
  echo "Selected CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} (free=${best_free_mib}MiB)"
fi

