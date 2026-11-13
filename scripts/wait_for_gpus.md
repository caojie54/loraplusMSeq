# Wait for available GPUs

Run in the environment needed by the task. The waiting script uses only the
Python standard library and `nvidia-smi`, not Torch. It polls every 60 seconds
by default and selects the requested number of GPUs with the most free memory.
The memory threshold is per GPU, in GiB (1024 MiB), not combined memory.

From `projects/loraplusMSeq`, wait for one GPU with at least 70 GiB free and
launch the supplied Qwen3.5 MedReason SGLang task:

```bash
conda activate mlora_sglang
python scripts/wait_for_gpus.py --num-gpus 1 --min-free-gb 70 \
  -- bash scripts/serve_medreason_qwen3_5_4b.sh
```

To wait in the background and keep running after the terminal closes:

```bash
nohup python -u scripts/wait_for_gpus.py --num-gpus 1 --min-free-gb 70 \
  -- bash scripts/serve_medreason_qwen3_5_4b.sh > wait_sglang0.log 2>&1 &
```

Wait progress goes to `wait_sglang0.log`; the server task writes to
`sglang0.log` in the working directory. Both paths can be absolute. Use
`LOG_NUM`, `MODEL_PATH`, `PORT`, `LOG_FILE`, `MEM_FRACTION_STATIC`, `DP_SIZE`,
and `TP_SIZE` environment variables to customize the server task. Single-GPU
defaults are `DP_SIZE=1`, `TP_SIZE=1`; a DP size of zero is not appropriate.

Example: two GPUs, restricted to indices 1, 5, 6, 7, two model replicas:

```bash
DP_SIZE=2 python scripts/wait_for_gpus.py --num-gpus 2 --min-free-gb 70 \
  --gpu-ids 1,5,6,7 -- bash scripts/serve_medreason_qwen3_5_4b.sh
```

Other options:

- `--interval 60`: seconds between queries.
- `--name-contains A100`: filter GPU model names.
- `--max-utilization 10`: also require utilization <= 10 percent.
- `--device-format uuid`: export UUIDs instead of numeric indices, avoiding
  differences between CUDA and nvidia-smi device enumeration.

Without `--gpu-ids`, candidate devices come from `CUDA_VISIBLE_DEVICES`, then
`SLURM_JOB_GPUS`, or all GPUs if neither is set. An explicitly empty visibility
mask is rejected. `--gpu-ids` overrides the inherited candidate list. Indices
refer to nvidia-smi indices; numeric mode sets `CUDA_DEVICE_ORDER=PCI_BUS_ID`.
MIG instances are not supported by this physical-GPU query.

Any command can follow `--`, for example `-- bash /path/to/task.sh`.
The task retains the caller's working directory and environment, with the
chosen `CUDA_VISIBLE_DEVICES` and `REQUIRED_FREE_GB=0` (existing MedReason
launchers skip reselection). The task must not overwrite CUDA_VISIBLE_DEVICES.
The waiter replaces itself with the task once; it propagates the task's exit
status and does not restart failed tasks. Ctrl-C cancels waiting.

This is a memory availability check, not an exclusive reservation: another
process can claim memory between the query and task startup. Query errors
are logged and retried; a missing nvidia-smi executable fails immediately.
