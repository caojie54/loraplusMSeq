# MedReason token length statistics

Run from `projects/loraplusMSeq`:

```bash
conda run --no-capture-output -n mlora_sglang python scripts/medreason_token_lengths.py --local-files-only
```

The installed environment is `mlora_sglang` (not `molora_sglang`). The script
loads only the cached `Qwen/Qwen3.5-4B` tokenizer; no GPU or model weights are
required. Use `--help` to override the dataset, tokenizer, cache, batch size,
and output directory.

## Training format

This reproduces the Qwen branch of `MedReason/src/model_training/SFT.py`,
specifically `Train_dataset.get_response()` and `get_prompt()`. For
`ours_quality_33000.jsonl`, the user message is `question`; the assistant is:

```text
## Thinking

{reasoning}

## Final Answer

{answer}
```

The `options` field is not appended separately. The tokenizer's chat template
is rendered with plain Jinja, exactly as in SFT.py. Full conversations use
`add_generation_prompt=False`; user-only prompts use `True`. Encoding uses
`add_special_tokens=False`. Full lengths include template tokens, before
truncation and batch padding. SFT.py masks the first `len(query_ids)` positions
and then keeps the last `max_seq_len` tokens (default 8192).

The Qwen3.5 template inserts an empty `<think>...</think>` section in the full
assistant message while the generation prompt ends with an open `<think>`.
The tokenized query is not an exact prefix of the full sequence in these
32,682 samples. `supervised_position_count` reproduces SFT.py's positional
masking; it is not a standalone token count of the answer text, nor does it
account for the causal LM label shift. This script does not alter training.

## Results

Measured in `mlora_sglang` with Python 3.12.16 and Transformers 5.19.0:

| Full sequence metric | Tokens |
| --- | ---: |
| Samples | 32,682 |
| Mean | 749.15 |
| Minimum / maximum | 265 / 4,627 |
| P25 / P50 / P75 | 630 / 735 / 850 |
| P90 / P95 | 965 / 1041.95 |
| P99 / P99.9 | 1209 / 1665.828 |

Percentiles use linear interpolation. Mean prompt length is 84.39 tokens;
mean supervised position count is 664.76. Sequences exceeding 1536, 2048,
4096, and 8192 tokens number 47, 11, 1, and 0 respectively.

Outputs default to
`/data/user/cj/experiments/medreason_token_lengths_qwen3.5_4b/`:

- `summary.json`: distributions, truncation rates, source counts, versions,
  and dataset/template hashes.
- `lengths.csv`: per-sample lengths keyed by original JSONL line number.
- `chat_template.jinja`: exact template used in the run.

Five samples were compared against the original SFT.py methods: full token
IDs, supervised counts, and left truncation all matched.
