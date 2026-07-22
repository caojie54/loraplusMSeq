"""Generate HumanEval pass@k samples for CodeAlpaca-style code tuning."""

import argparse
import json
import os
import time

import torch
import transformers
from datasets import Dataset
from tqdm import tqdm
from transformers import GenerationConfig, TextGenerationPipeline
from transformers.pipelines.pt_utils import KeyDataset

from generation_test_utils import load_model_and_tokenizer, supports_use_model_defaults
from loraplusmseq.data import get_formatted_datasets


def str2bool(value):
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"true", "1", "yes", "y", "on"}:
        return True
    if lowered in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run HumanEval generation with 10 sampled completions per prompt.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument(
        "--data_path",
        type=str,
        default="/mnt/petrelfs/caojie1/projects/CoMoL/datasets/eval_code",
    )
    parser.add_argument("--max_new_tokens", type=int, default=400)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_return_sequences", type=int, default=10)
    parser.add_argument("--do_sample", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trust_remote_code", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    transformers.set_seed(0)

    dtype = torch.bfloat16 if args.bf16 else torch.float16
    model, tokenizer, resolved_model_path = load_model_and_tokenizer(
        args.model_path,
        dtype=dtype,
        trust_remote_code=args.trust_remote_code,
    )
    print(f"Model loaded from {resolved_model_path}")

    output_root = args.output_dir or args.model_path
    prediction_dir = os.path.join(output_root, "predictions")
    os.makedirs(prediction_dir, exist_ok=True)
    print(f"Predictions will be saved to {prediction_dir}")

    generation_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        num_return_sequences=args.num_return_sequences,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    print(f"Generation config: {generation_config}")
    pipeline = TextGenerationPipeline(model=model, tokenizer=tokenizer)

    benchmark = "humaneval"
    split = "test"
    data_path = os.path.join(args.data_path, benchmark)
    formatted_datasets = get_formatted_datasets(data_path=data_path, prompt_only=True)
    dataset = formatted_datasets[split]
    if args.max_samples > 0:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    start_time = time.time()
    responses = []
    generation_kwargs = {}
    if supports_use_model_defaults():
        generation_kwargs["use_model_defaults"] = False
    for response in tqdm(
        pipeline(
            KeyDataset(dataset, "text"),
            generation_config=generation_config,
            return_full_text=False,
            batch_size=args.batch_size,
            **generation_kwargs,
        ),
        total=len(dataset),
    ):
        responses.append([res["generated_text"] for res in response])

    elapsed = time.time() - start_time
    print(f"Time taken for {benchmark}: {elapsed:.2f} seconds")
    if responses:
        print(f"Response example:\n{responses[0]}")

    rows = []
    for row, response in zip(dataset, responses):
        output_row = dict(row)
        output_row["response"] = response
        rows.append(output_row)

    Dataset.from_list(rows).to_json(
        os.path.join(prediction_dir, f"{benchmark}_responses.jsonl"),
        lines=True,
    )
    with open(os.path.join(prediction_dir, f"{benchmark}_time.txt"), "w", encoding="utf-8") as handle:
        handle.write(f"{elapsed:.2f}")


if __name__ == "__main__":
    main()
