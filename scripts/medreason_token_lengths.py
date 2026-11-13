"""Measure MedReason lengths using the Qwen branch of MedReason/SFT.py.

Uses its exact Jinja rendering and add_special_tokens=False, without truncation
or padding. No model weights or GPU are needed. Prompt masking is positional,
as in SFT.py; supervised lengths include template-generated assistant tokens.
"""

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import jinja2
import numpy as np
import transformers
from jinja2 import Template
from transformers import AutoTokenizer


def render_training_pair(template, row):
    for key in ("question", "reasoning", "answer"):
        if not isinstance(row.get(key), str):
            raise ValueError(f"Expected string field {key!r}")
    response = "## Thinking\n\n{}\n\n## Final Answer\n\n{}".format(
        row["reasoning"], row["answer"]
    )
    user = {"role": "user", "content": row["question"]}
    full = template.render(
        messages=[user, {"role": "assistant", "content": response}],
        add_generation_prompt=False,
    )
    prompt = template.render(messages=[user], add_generation_prompt=True)
    return full, prompt


def summarize(values):
    values = np.asarray(values, dtype=np.int64)
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9]
    return {
        "count": int(values.size),
        "total_tokens": int(values.sum()),
        "mean": float(values.mean()),
        "std_population": float(values.std()),
        "min": int(values.min()),
        "max": int(values.max()),
        "percentiles": dict(zip(
            (f"p{p:g}" for p in percentiles),
            np.percentile(values, percentiles, method="linear").tolist(),
        )),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=Path(
        "/data/user/cj/datasets/MedReason/ours_quality_33000.jsonl"))
    parser.add_argument("--tokenizer", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--cache-dir", default="/data/user/cj/hf/hub")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output-dir", type=Path, default=Path(
        "/data/user/cj/experiments/medreason_token_lengths_qwen3.5_4b"))
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
    )
    if not isinstance(tokenizer.chat_template, str):
        raise ValueError("A single tokenizer chat_template string is required")
    # Deliberately match SFT.py instead of changing its rendering semantics.
    template = Template(tokenizer.chat_template)
    lengths, sources = [], Counter()
    nonprefix_count = 0
    batch = []

    def process_batch():
        nonlocal nonprefix_count
        pairs = [render_training_pair(template, row) for _, row in batch]
        full_ids = tokenizer(
            [pair[0] for pair in pairs], add_special_tokens=False,
            truncation=False, padding=False, return_attention_mask=False,
        )["input_ids"]
        prompt_ids = tokenizer(
            [pair[1] for pair in pairs], add_special_tokens=False,
            truncation=False, padding=False, return_attention_mask=False,
        )["input_ids"]
        for (line, row), full, prompt in zip(batch, full_ids, prompt_ids):
            if len(prompt) > len(full):
                raise ValueError(f"Line {line}: prompt longer than full sequence")
            nonprefix_count += full[:len(prompt)] != prompt
            lengths.append((line, len(full), len(prompt), len(full) - len(prompt)))
            sources[row.get("dataset_name", "unknown")] += 1
        print(f"Processed {len(lengths)} samples", file=sys.stderr)
        batch.clear()

    with args.data_path.open(encoding="utf-8") as handle:
        for line, text in enumerate(handle, 1):
            if not text.strip():
                continue
            batch.append((line, json.loads(text)))
            if len(batch) == args.batch_size:
                process_batch()
    if batch:
        process_batch()
    if not lengths:
        raise ValueError("Dataset is empty")

    matrix = np.asarray(lengths, dtype=np.int64)
    full = matrix[:, 1]
    report = {
        "data_path": str(args.data_path.resolve()),
        "data_sha256": hashlib.sha256(args.data_path.read_bytes()).hexdigest(),
        "tokenizer": args.tokenizer,
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_commit": tokenizer.init_kwargs.get("_commit_hash"),
        "chat_template_sha256": hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
        "python": sys.version,
        "python_executable": sys.executable,
        "transformers": transformers.__version__,
        "jinja2": jinja2.__version__,
        "numpy": np.__version__,
        "format": "MedReason SFT.py Qwen branch; question + assistant Thinking/Final Answer; no extra options",
        "length_semantics": "Before truncation/padding; no added special tokens; quantiles use linear interpolation",
        "prompt_not_token_prefix_count": nonprefix_count,
        "source_counts": dict(sources),
        "full_sequence": summarize(full),
        "prompt": summarize(matrix[:, 2]),
        "supervised_position_count": summarize(matrix[:, 3]),
        "sequence_limits": {},
    }
    for limit in (1536, 2048, 4096, 8192, 16384, 32768):
        report["sequence_limits"][str(limit)] = {
            "exceed_count": int((full > limit).sum()),
            "exceed_percent": float((full > limit).mean() * 100),
            "mean_after_left_truncation": float(np.minimum(full, limit).mean()),
            "discarded_tokens": int(np.maximum(full - limit, 0).sum()),
        }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "chat_template.jinja").write_text(tokenizer.chat_template, encoding="utf-8")
    with (args.output_dir / "lengths.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["line_number", "full_tokens", "prompt_tokens", "supervised_tokens"])
        writer.writerows(lengths)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
