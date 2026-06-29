# loraplusMSeq

Sequential LoRA plus selected compensation-module training. The project supports `commonsense170k` and NaturalReasoning scripts.

For each n-batch, the trainer first updates LoRA parameters. It then selects original base modules and replays the same cached n-batch with LoRA frozen and only the selected original module weights trainable.

## Selection methods

- `alpha`: select modules by accumulated LoRA gradient pressure. Use `--alpha_score lora_grad_norm`, `lora_grad_norm_min`, or `lora_update_ratio`.
- `static_random`: sample one fixed random module set and reuse it for all module replay phases.
- `dynamic_random`: sample a fresh random module set after every LoRA n-batch.
- `lora`: LoRA-only baseline; module replay is skipped.

`--compensation_ratio` is interpreted against total original model parameters, not total candidate-module parameters. The run metadata still records `selected_candidate_param_ratio` for comparison.

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

- train data: `/mnt/petrelfs/caojie1/projects/CoMoL/datasets/natural_reasoning_20k`
- eval data: `/mnt/petrelfs/caojie1/projects/CoMoL/datasets/natural_reasoning_eval`
- eval benchmarks: `gpqa_diamond math_500 mmlu_pro_500`
- test generation: `max_new_tokens=1536`, `batch_size=64`

## Outputs

Runs write under `/mnt/dhwfile/raise/user/caojie/loraplusMSeq/outputs` by default. Each run records:

- `candidate_modules.json`: candidate modules, total model parameter denominator, and candidate parameter totals.
- `selection_history.jsonl`: selected modules and selected ratios per block.
- `trainable_params.json`: LoRA/module trainable parameter stats, optimizer strategy, optimizer dtype, gradient mode, and ratio denominator.
- `train_metrics.json`: training loss/runtime and the same key optimizer/state metadata.
- `predictions/` and `natural_reasoning_acc_score.jsonl` for NaturalReasoning runs when evaluation is enabled.
