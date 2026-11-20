"""Measure NaturalReasoning sequence lengths using train.py's formatting and EOS rules."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import transformers
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from loraplusmseq.data import format_text


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
    parser.add_argument("--data-path", type=Path, default=ROOT / "datasets/natural_reasoning_20k/train.jsonl")
    parser.add_argument("--tokenizer", default="google/gemma-4-12B")
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/token_lengths/natural_reasoning_20k_gemma4_12b")
    args = parser.parse_args()
    if args.batch_size <= 0 or args.max_length <= 0:
        parser.error("batch-size and max-length must be positive")
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, padding_side="left", trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    if tokenizer.eos_token_id is None:
        raise ValueError("The tokenizer must define eos_token_id to follow the training EOS rule.")
    lengths = []
    batch = []
    digest = hashlib.sha256()

    def process_batch():
        texts = [format_text(row, "natural_reasoning_20k", prompt_only=False)["text"] for _, row in batch]
        full_ids = tokenizer(texts, truncation=False, padding=False, return_attention_mask=False)["input_ids"]
        train_ids = tokenizer(
            texts, truncation=True, max_length=args.max_length,
            padding=False, return_attention_mask=False,
        )["input_ids"]
        for (line, _), full, train in zip(batch, full_ids, train_ids):
            full_eos_added = bool(full) and full[-1] != tokenizer.eos_token_id
            train_eos_added = bool(train) and train[-1] != tokenizer.eos_token_id and len(train) < args.max_length
            lengths.append((line, len(full), len(full) + int(full_eos_added),
                            len(train) + int(train_eos_added), int(train_eos_added),
                            int(len(full) > args.max_length)))
        print(f"Processed {len(lengths)} samples", file=sys.stderr, flush=True)
        batch.clear()

    with args.data_path.open("rb") as handle:
        for line_number, line in enumerate(handle, 1):
            digest.update(line)
            if not line.strip():
                continue
            batch.append((line_number, json.loads(line)))
            if len(batch) == args.batch_size:
                process_batch()
    if batch:
        process_batch()
    if not lengths:
        raise ValueError("The dataset is empty")

    matrix = np.asarray(lengths, dtype=np.int64)
    full = matrix[:, 2]
    report = {
        "data_path": str(args.data_path.resolve()),
        "data_sha256": digest.hexdigest(),
        "tokenizer": args.tokenizer,
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_commit": tokenizer.init_kwargs.get("_commit_hash"),
        "tokenizer_backend_sha256": hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest() if tokenizer.is_fast else None,
        "transformers": transformers.__version__,
        "python_executable": sys.executable,
        "max_length": args.max_length,
        "truncation_side": tokenizer.truncation_side,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "format": "loraplusmseq.data.format_text(prompt_only=False): instruction/input template + output; tokenizer default special tokens; no chat template",
        "length_semantics": "unpadded_full includes EOS when missing; training applies tokenizer truncation first, then adds missing EOS only if length < max_length. Lengths exclude batch padding. Percentiles use linear interpolation.",
        "tokenized_before_eos": summarize(matrix[:, 1]),
        "unpadded_full": summarize(full),
        "training": summarize(matrix[:, 3]),
        "training_truncated_count": int(matrix[:, 5].sum()),
        "training_truncated_percent": float(matrix[:, 5].mean() * 100),
        "training_eos_appended_count": int(matrix[:, 4].sum()),
        "sequence_limits": {},
    }
    for limit in sorted({args.max_length, 1024, 1200, 1536, 2048, 4096, 8192, 16384}):
        report["sequence_limits"][str(limit)] = {
            "full_exceed_count": int((full > limit).sum()),
            "full_exceed_percent": float((full > limit).mean() * 100),
        }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "lengths.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["line_number", "tokenized_before_eos", "unpadded_full_tokens",
                         "training_tokens", "training_eos_appended", "training_truncated"])
        writer.writerows(lengths)
    (args.output_dir / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Saved results to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
