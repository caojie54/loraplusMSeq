# loraplusMSeq

Sequential LoRA plus selected compensation-module training. The project supports `commonsense170k` and NaturalReasoning scripts.

For each n-batch, the trainer first updates LoRA parameters. It then selects original base modules and replays the same cached n-batch with LoRA frozen and only the selected original module weights trainable.

## Installation

```bash
# create enviroment
conda create -n mlora python==3.12

# Install required dependencies
pip install -r requirements.txt

# Install sglang for med eval
pip install uv
# 根据sglang 官方文档，安装了，后续med eval只需要额外安装https；
uv pip install --prerelease=allow sglang

pip install flash-linear-attention

```

## Selection methods

- `alpha`: select modules by accumulated LoRA gradient pressure. Use `--alpha_score lora_grad_norm`, `lora_effective_update_pressure`, or `lora_update_ratio`. The variants `lora_grad_norm_min` and `lora_effective_update_pressure_min` keep the same raw scores but prefer smaller values. With candidate-pool sampling enabled, they use `-log(score)` before the existing size correction, normalization, and sampling. For example, set `METHOD=alpha ALPHA_SCORE=lora_effective_update_pressure_min` when running `bash task_medreason.sh` or `bash task_natural_reasoning.sh`.
- `static_random`: sample one fixed random module set and reuse it for all module replay phases.
- `dynamic_random`: sample a fresh random module set after every LoRA n-batch.
- `lora`: LoRA-only baseline; module replay and module scoring are skipped. Gradient scores are computed only for `alpha`; random methods also skip scoring.

`--compensation_ratio` is interpreted against total original model parameters, not total candidate-module parameters. The run metadata still records `selected_candidate_param_ratio` for comparison.

## NaturalReasoning-20k static random runs

Each launcher below runs training, saves `merged/`, generates predictions, and
scores `gpqa_diamond`, `math_500`, and `mmlu_pro_500`. Replace `seed0` with `seed1`
or `seed2` to select the other training seeds:

```bash
bash exps/natural_reasoning_20k/finetune_llama_seq_static_random_rank64_ratio0.02_block50_seed0.sh
bash exps/natural_reasoning_20k_gemma3/finetune_gemma3_4b_pt_seq_static_random_rank64_ratio0.02772_block50_seed0.sh
```

Both use rank 64, block size 50, one LoRA epoch plus module replay, length 1536,
batch size 4 with accumulation 8, LoRA LR `1e-4`, module LR `1e-5`, BF16 LoRA
optimizer, and FP32 module optimizer with `reset_offload`. Llama uses ratio
`0.02` and evaluation batch 64; Gemma uses `0.02772` and evaluation batch 120.
The sampled module set stays fixed throughout each run. Generation keeps the
existing greedy evaluation defaults; the three seeds control training/selection.

Launchers use the active Python environment and local dataset paths. They can
be invoked from any working directory. `CUDA_VISIBLE_DEVICES` restricts GPU
candidates; `REQUIRED_FREE_GB=0` retains the caller's GPU assignment. Outputs
default to `.../outputs/natural-reasoning-20k/<model-tag>/<run-name>/`, with the
training seed in each default run name. `OUTPUT_ROOT` and `RUN_NAME` can override
these locations.

## Optimizer state

`--module_optimizer_state_strategy` supports:

- `reset_offload` (default): recreate the module optimizer every module replay block and release/offload module optimizer state before returning to LoRA.
- `persistent_offload`: keep one persistent module optimizer over all candidate original modules. Optimizer state is restored for module replay and offloaded around LoRA phases.

LoRA optimizer reset controls are unchanged:

- `--lora_optimizer_reset_strategy keep`
- `--lora_optimizer_reset_strategy reset_all`
- `--lora_optimizer_reset_strategy reset_selected`

## Optimizer update dtype

Both phases can update either the low-precision model parameters directly or fp32 shadow parameters:

- `--lora_optimizer_dtype bf16|fp32` (default `bf16`)
- `--module_optimizer_dtype bf16|fp32` (default `bf16`)

`fp32` uses `Fp32ShadowAdamW`: gradients are copied to fp32 shadow parameters, AdamW updates the fp32 shadows, then updated values are copied back to the model parameter dtype.

## Module gradient mode

`--module_gradient_mode full|residual` controls module replay gradients.

- `full` (default): use the full selected base-module gradient.
- `residual`: project selected base weight gradients into the residual subspace orthogonal to the current LoRA update bases, following the implementation in `loraplusMSeqRe`. Tune numerical rank with `--residual_rtol`.

## Entry points

Commonsense:

```bash
cd /mnt/petrelfs/caojie1/projects/loraplusMSeq
srun -p sciverse_agent --job-name=mseq_debug --ntasks-per-node=1 --cpus-per-task=24 --gres=gpu:1 bash task.sh
```

NaturalReasoning 20k:

```bash
cd /mnt/petrelfs/caojie1/projects/loraplusMSeq
srun -p sciverse_agent --job-name=nr20k_mseq --ntasks-per-node=1 --cpus-per-task=24 --gres=gpu:1 bash task_natural_reasoning.sh
```

NaturalReasoning task defaults:

- train data: `/data/user/cj/projects/loraplusMSeq/datasets/natural_reasoning_20k`
- eval data: `/data/user/cj/projects/loraplusMSeq/datasets/natural_reasoning_eval`
- eval benchmarks: `gpqa_diamond math_500 mmlu_pro_500`
- test generation: `max_new_tokens=1536`, `batch_size=64`

## Outputs

Runs write under `/mnt/dhwfile/raise/user/caojie/loraplusMSeq/outputs` by default. Each run records:

- `candidate_modules.json`: candidate modules, total model parameter denominator, and candidate parameter totals.
- `selection_history.jsonl`: selected modules and selected ratios per block.
- `trainable_params.json`: LoRA/module trainable parameter stats, optimizer strategy, optimizer dtype, gradient mode, and ratio denominator.
- `train_metrics.json`: training loss/runtime and the same key optimizer/state metadata.
- `predictions/` and `natural_reasoning_acc_score.jsonl` for NaturalReasoning runs when evaluation is enabled.

## Training memory statistics

`memory_log.jsonl` records **every optimizer update**, including the final partial
logging interval. Phases are `lora` / `module` in loraplusMSeq and `misa` in MISA.
Both projects use the same counters and sampling rules:

- `peak_memory_gib`: allocated-memory peak of this update, with CUDA peak counters
  reset at its start; this includes the live model/optimizer baseline.
- `max_memory_gib`: maximum update peak observed so far.
- `avg_peak_memory_gib`: arithmetic mean of update peaks over **all
  completed updates**, not averages of cumulative maxima or of printed logs.
- `memory_samples`: number of updates included in the averages.

`train_log.jsonl` includes overall running statistics and `phase_memory` for the
current phase. `train_metrics.json` includes overall statistics and
`memory_by_phase` summaries. Overall means are weighted by update count, not by
phase count or wall time. Current allocated memory and its average are not
queried or recorded.

These are PyTorch **allocated** bytes divided by `1024**3` on the model's input
device (`memory_device`), not allocator-reserved memory, total GPU usage, or a
sum over multiple GPUs. CPU execution records zero. Counters are read without
adding CUDA synchronizations or emptying the allocator cache.

Measurements exclude model loading, validation, saving, and transitions outside
updates. LoRA state restoration is inside its update; module selection/optimizer
setup is included in the first replay update. MISA's optimizer-step module switch
is included in that update. Existing persistent model/optimizer tensors still
count toward every update's absolute footprint. These training-only maxima
replace the old process-lifetime `max_memory_gib` and are not directly comparable
to old logs. Clearing temporary forward outputs also avoids retaining previous
microbatch logits during the next forward pass.

## Dataset locations and MedReason result directory

NaturalReasoning training and evaluation data default to
`/data/user/cj/projects/loraplusMSeq/datasets/natural_reasoning_20k` and
`/data/user/cj/projects/loraplusMSeq/datasets/natural_reasoning_eval` in both
loraplusMSeq and MISA launchers. MedReason evaluation data and code are under
`/data/user/cj/projects/loraplusMSeq/MedReason/{eval_data,evaluation}`.
MedReason training data remains in `/data/user/cj/datasets/MedReason`.

The evaluation CLI accepts `--results_dir /path/to/results` and writes
`<results_dir>/<task_floder>/logs/` and `<results_dir>/<task_floder>/result/`.
The default is `./results`, relative to the working directory of `eval.py`.
The wrapper changes directory to `MEDREASON_ROOT` and accepts `RESULTS_DIR`:

```bash
MODEL_DIR=/path/to/run/merged PORT=28035 RESULTS_DIR=/path/to/eval-results \
  bash /data/user/cj/projects/loraplusMSeq/eval_med_shell/eval_medreason.sh
```

SGLang must already be serving the specified model. With no override, wrapper
results are saved under `loraplusMSeq/MedReason/results/`.
