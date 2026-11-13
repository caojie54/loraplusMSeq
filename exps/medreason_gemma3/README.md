# Gemma3 4B PT LoRA on MedReason

Run from `projects/loraplusMSeq` in the training environment:

```bash
conda activate mlora_sglang
bash exps/medreason_gemma3/finetune_gemma3_4b_pt_lora_rank64_epoch2_medreason_seed0.sh
```

Defaults follow the NaturalReasoning LoRA experiment: rank 64, 2 epochs,
seed 0, bf16 optimizer, learning rate 1e-4, batch size 4, accumulation 8,
warmup 50, and all seven attention/MLP projection targets. The current LoRA
launcher defaults to length 1200; the sequential presets below use 1536.
The launcher chooses the candidate GPU with the most free memory, requiring
75 GiB. Set `CUDA_VISIBLE_DEVICES=1,2` to restrict the candidates. It uses the
currently active Python environment and does not activate Conda internally.

`task_medreason.sh` calls `train.py --data_format medreason`. The JSONL file
defaults to `/data/user/cj/datasets/MedReason/ours_quality_33000.jsonl`.
The prompt is `question + "\n\n"`; the response is
`## Thinking\n\n{reasoning}\n\n## Final Answer\n\n{answer}`. No chat template,
Alpaca wrapper, or additional options are inserted. Prompt tokens are masked
from loss. EOS is appended to the response. Overlong sequences are truncated
from the left, as in MedReason's SFT code; this can remove question context.
The existing NaturalReasoning path remains the default in `train.py`.

Both `adapter/` and `merged/` are saved under
`/data/user/cj/experiments/loraplusMSeq/outputs/medreason/gemma-3-4b-pt/<RUN_NAME>/`.
The merged model is required by the evaluation workflow below. Override
`OUTPUT_ROOT`, `RUN_NAME`, `MAX_LENGTH`, or `REQUIRED_FREE_GB` through environment
variables. Training does not start an evaluation server automatically.

## Evaluate with MedReason

After training, start SGLang in `mlora_sglang`, using the actual merged model:

```bash
export MODEL_DIR=/data/user/cj/experiments/loraplusMSeq/outputs/medreason/gemma-3-4b-pt/gemma-3-4b-pt-lora-qkvogateupdown-rank64-medreason-epoch2-seed0-loraoptbf16-loralr1e-4/merged
CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server \
  --model-path "$MODEL_DIR" --port 28035
```

Choose an available GPU and an unused port. In another terminal, activate
`mlora_sglang`, set the same `MODEL_DIR`, and run from `loraplusMSeq`:

```bash
PORT=28035 MODEL_DIR="$MODEL_DIR" bash eval_med_shell/eval_medreason.sh
```

This invokes `MedReason/src/evaluation/eval.py` for medbullets_op4,
medbullets_op5, HLE_biomed, medqa_test, and MedXpertQA_test, with
`--strict_prompt --no-use_chat_template`. It preserves MedReason's plain-text
multiple-choice evaluation prompt (different wording from the training
questions). Defaults are batch size 120 and 2000 new tokens; override with
`TEST_BATCH_SIZE` and `TEST_MAX_NEW_TOKENS`.

The server must already serve the trained model: `--model_name` in MedReason
only names the evaluation run when chat templates are disabled; it does not
switch server weights. Results go to `MedReason/results/Med_eval/<RUN_NAME>/`.
The local MedReason scorer already supports MedXpertQA's `label` field.

## Sequential module training

`task_medreason.sh` supports `METHOD=lora|static_random|dynamic_random|alpha`.
LoRA defaults to 2 epochs; the other methods default to 1 epoch of LoRA
updates plus replay of the same data on selected base modules. Epoch counts
can be overridden with `NUM_TRAIN_EPOCHS`.

Gemma3 presets in this directory:

- `finetune_gemma3_4b_pt_seq_static_random_rank64_ratio0.02772_block50_seed0.sh`
- `finetune_gemma3_4b_pt_seq_dynamic_random_rank64_ratio0.02772_block50_seed0.sh`
- `finetune_gemma3_4b_pt_seq_alpha_grad_norm_rank64_ratio0.02772_candidate0.08316_block50_seed0.sh`
- `finetune_gemma3_4b_pt_seq_alpha_effective_update_pressure_rank64_ratio0.02772_candidate0.08316_block50_seed0.sh`

All use rank 64, module parameter budget 0.02772 of the original model,
selection interval 50 LoRA optimizer updates, LoRA LR 1e-4, module LR 1e-5,
bf16 LoRA optimizer, and FP32 module optimizer with `reset_offload`.
Static random reuses one selection; dynamic random selects every block.
Both alpha presets use candidate budget 0.08316, temperature 1.0, uniform
mixture 0.1, score gamma 1.0, and no group normalization.

For example, from the repository root:

```bash
bash exps/medreason_gemma3/finetune_gemma3_4b_pt_seq_alpha_effective_update_pressure_rank64_ratio0.02772_candidate0.08316_block50_seed0.sh

# Direct ranking rather than candidate-pool sampling:
METHOD=alpha ALPHA_SCORE=lora_grad_norm ALPHA_CANDIDATE_RATIO=0 bash task_medreason.sh

# Other scores supported by the same trainer:
METHOD=alpha ALPHA_SCORE=lora_grad_norm_min bash task_medreason.sh
METHOD=alpha ALPHA_SCORE=lora_update_ratio bash task_medreason.sh
```

Available overrides match the NaturalReasoning launcher:
`MODULE_LR`, `COMP_RATIO`, `SELECTION_INTERVAL`, `ALPHA_SCORE`,
`ALPHA_CANDIDATE_RATIO`, `ALPHA_SAMPLING_TEMPERATURE`, `ALPHA_UNIFORM_MIX`,
`ALPHA_SCORE_GAMMA`, `ALPHA_GROUP_NORM`, `LORA_OPTIMIZER_RESET_STRATEGY`,
`MODULE_OPTIMIZER_STATE_STRATEGY`, `MODULE_OPTIMIZER_DTYPE`,
`MODULE_GRADIENT_MODE`, and `RESIDUAL_RTOL`.

Automatic run names distinguish method, score, candidate ratio, compensation
ratio, block size, seed, and optimizer dtype/state strategy. Set a distinct
`RUN_NAME` when comparing other overrides (for example sampling temperature,
group normalization, or residual gradients) to avoid reusing output paths.
The sequential presets save merged models for the same MedReason evaluation
workflow above; training and evaluation continue to use plain text.
