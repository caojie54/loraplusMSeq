# Gemma-4-12B MedReason training

Run from the repository root in the `mlora_sglang` environment. These five
launchers call `task_medreason.sh` with `google/gemma-4-12B`, rank 64 and seed 0.

| Method | Epochs | Module score | Candidate ratio |
| --- | --- | --- | --- |
| LoRA | 2 | — | — |
| Sequential alpha effective update pressure | 1 | lora_effective_update_pressure | 0.065789545788 |
| Sequential alpha gradient norm | 1 | lora_grad_norm | 0 |
| Dynamic random | 1 | — | 0 |
| Static random | 1 | — | 0 |

Sequential launchers use `COMP_RATIO=0.021929848596` and switch every 50 LoRA
updates. They replay each block on selected base modules.

## Parameter budget and trainable modules

Verified with Transformers 5.19.0, cached snapshot
`1a2c7729392586f5ecddae8807ee2ef741eabb9a`, and the actual functions in
`train.py` and `loraplusmseq/module_selection.py`.

The loaded class is `Gemma4UnifiedForConditionalGeneration`. Its language model
has 48 decoder layers, hidden size 3840, intermediate size 15360 and vocabulary
262144. Forty layers use sliding attention and eight use full attention.
Full-attention layers reuse K as V, giving 40 independent `v_proj` modules.

The target resolver selects 328 language-model projections. LoRA updates only
their adapter weights. Module replay updates selected original projection
weights. Vision/audio projections, embeddings, norms and the tied LM head
remain frozen.

| Parameter group | Count |
| --- | --- |
| Language model, including tied embeddings | 11,907,350,272 |
| Vision projections and embeddings | 49,922,304 |
| Audio projection | 2,457,600 |
| Original model total, used as ratio denominator | 11,959,730,176 |
| Rank 64 LoRA | 262,275,072 |

For each targeted linear layer, LoRA has `64 * (in_features + out_features)`
parameters. The total is `40 * 5,455,872 + 8 * 5,505,024`.
Dividing by the original model total gives `0.02192984859527319`.
The configured ratio rounds upward to preserve the integer budget of 262,275,072.

Whole modules have discrete sizes. Seed 0 random selection uses 261,488,640
parameters, approximately 0.30% below the budget. Actual per-block counts are
recorded in `selection_history.jsonl`.
The complete structure and target names are in
[gemma_4_12b_structure.txt](../../gemma_4_12b_structure.txt).

## Run and configuration

```bash
conda activate mlora_sglang
bash exps/medreason_gemma4/finetune_gemma4_12b_lora_rank64_epoch2_medreason_seed0.sh
```

Defaults use length 1200, batch size 1, accumulation 32 and gradient checkpointing.
This retains the reference effective batch size of 32 while reducing activation
memory. LoRA learning rate is 1e-4 with bf16 optimizer state. Module learning rate
is 1e-5 with fp32 optimizer state and reset/offload.

GPU selection requires 70 GiB free. Override `CUDA_VISIBLE_DEVICES`,
`REQUIRED_FREE_GB`, `SEED` or batch settings as needed. Changing rank or model
requires recalculating the equivalent ratio. Data defaults to
`/data/user/cj/datasets/MedReason/ours_quality_33000.jsonl`; output defaults to
`/data/user/cj/data2/experiments/loraplusMSeq/outputs/medreason/gemma-4-12b`.

Validation covered full-size meta-model parameter accounting, launcher argument
forwarding, and forward/backward passes on a reduced Gemma4Unified model.
Full 12B GPU training and its peak memory have yet to be verified.
