# Llama-3.1-8B MedReason experiments

These presets use the base model `meta-llama/Llama-3.1-8B`, seed 0, and the
plain-text MedReason adapter in `task_medreason.sh`. No chat template is used;
only reasoning/answer tokens contribute to the loss.

Run from `projects/loraplusMSeq`:

```bash
conda activate mlora_sglang

# LoRA: two epochs.
bash exps/medreason_llama3_1/finetune_llama3_1_8b_lora_rank64_epoch2_medreason_seed0.sh

# Sequential: one epoch of LoRA updates plus selected-module replay.
bash exps/medreason_llama3_1/finetune_llama3_1_8b_seq_alpha_effective_update_pressure_rank64_ratio0.02772_candidate0.08316_block50_seed0.sh
```

Common defaults follow the current Gemma MedReason scripts: rank 64, maximum
length 1200, batch size 4, gradient accumulation 8, LoRA LR 1e-4, bf16 LoRA
optimizer, and gradient checkpointing disabled. GPU selection requires at
least 75 GiB free; restrict candidates with `CUDA_VISIBLE_DEVICES` if needed.
Use `GRADIENT_CHECKPOINTING=true` or reduce `TRAIN_BATCH_SIZE` if needed for
the larger model; the free-memory check is not a measured peak-memory bound.

Sequential defaults: effective update pressure, module ratio 0.02772,
candidate ratio 0.08316, block 50, module LR 1e-5, FP32 module optimizer,
and `reset_offload` state handling.

Both scripts save `adapter/` and `merged/` under
`/data/user/cj/data2/experiments/loraplusMSeq/outputs/medreason/llama-3-1-8b/<RUN_NAME>/`.
`OUTPUT_ROOT` and `RUN_NAME` can be overridden. For evaluation, start SGLang
with this run's `merged/` model, then invoke the existing wrapper:

```bash
MODEL_DIR=/absolute/path/to/run/merged PORT=28035 bash eval_med_shell/eval_medreason.sh
```

The evaluation wrapper uses MedReason's five benchmarks with chat templates
disabled. It does not start the server or change the model already served.
