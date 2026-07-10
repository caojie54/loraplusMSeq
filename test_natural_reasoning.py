"""Generate NaturalReasoning benchmark predictions for a trained model."""

import argparse
import os
import time
from typing import List

import torch
import transformers
from datasets import Dataset
from peft import PeftConfig, PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, TextGenerationPipeline
from transformers.pipelines.pt_utils import KeyDataset

from loraplusmseq.data import get_formatted_datasets

DEFAULT_DO_SAMPLE = False
DEFAULT_TEMPERATURE = 0.6
DEFAULT_TOP_P = 0.9


def str2bool(value):
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"true", "1", "yes", "y", "on"}:
        return True
    if lowered in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def add_generation_args(parser):
    parser.add_argument(
        "--do_sample",
        type=str2bool,
        nargs="?",
        const=True,
        default=None,
        help="Use sampling. Defaults to false/greedy when omitted; pass --do_sample true for sampling.",
    )
    parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature. Defaults to 0.6 when sampling is enabled.")
    parser.add_argument("--top_p", type=float, default=None, help="Nucleus sampling top-p. Defaults to 0.9 when sampling is enabled.")



def supports_use_model_defaults():
    try:
        return int(transformers.__version__.split(".", 1)[0]) < 5
    except (IndexError, ValueError):
        return True


def build_generation_config(args, tokenizer):
    do_sample = DEFAULT_DO_SAMPLE if args.do_sample is None else args.do_sample
    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": do_sample,
    }
    if tokenizer.pad_token_id is not None:
        generation_kwargs["pad_token_id"] = tokenizer.pad_token_id
    if tokenizer.eos_token_id is not None:
        generation_kwargs["eos_token_id"] = tokenizer.eos_token_id

    if do_sample:
        generation_kwargs["temperature"] = DEFAULT_TEMPERATURE if args.temperature is None else args.temperature
        generation_kwargs["top_p"] = DEFAULT_TOP_P if args.top_p is None else args.top_p
    else:
        if args.temperature is not None:
            generation_kwargs["temperature"] = args.temperature
        if args.top_p is not None:
            generation_kwargs["top_p"] = args.top_p

    return GenerationConfig(**generation_kwargs)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run GPQA-Diamond, MATH-500, and MMLU-Pro-500 generation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument(
        "--data_path",
        type=str,
        default="/mnt/petrelfs/caojie1/projects/CoMoL/datasets/natural_reasoning_eval",
    )
    parser.add_argument("--benchmarks", nargs="+", default=["gpqa_diamond", "math_500", "mmlu_pro_500"])
    parser.add_argument("--max_new_tokens", type=int, default=1536)
    parser.add_argument("--batch_size", type=int, default=64)
    add_generation_args(parser)
    parser.add_argument("--output_dir", type=str, default=None, help="Optional root directory for predictions; defaults to model_path for backward compatibility.")
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--trust_remote_code", action="store_true", default=True)
    return parser.parse_args()


def configure_tokenizer(tokenizer, source_path: str) -> None:
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is not None:
        return
    if "llama-3" in source_path.lower():
        tokenizer.pad_token_id = 128255
    else:
        tokenizer.pad_token = tokenizer.eos_token


def resolve_model_path(model_path: str) -> str:
    merged_path = os.path.join(model_path, "merged")
    if os.path.isdir(merged_path):
        return merged_path
    adapter_path = os.path.join(model_path, "adapter")
    if os.path.exists(os.path.join(adapter_path, "adapter_config.json")):
        return adapter_path
    return model_path


def load_model_and_tokenizer(model_path: str, dtype: torch.dtype, trust_remote_code: bool):
    resolved_model_path = resolve_model_path(model_path)
    if os.path.exists(os.path.join(resolved_model_path, "adapter_config.json")):
        peft_config = PeftConfig.from_pretrained(resolved_model_path)
        base_model_path = peft_config.base_model_name_or_path
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=trust_remote_code,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            padding_side="left",
            trust_remote_code=trust_remote_code,
        )
        configure_tokenizer(tokenizer, base_model_path)
        model = PeftModel.from_pretrained(base_model, resolved_model_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            resolved_model_path,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=trust_remote_code,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            resolved_model_path,
            padding_side="left",
            trust_remote_code=trust_remote_code,
        )
        configure_tokenizer(tokenizer, resolved_model_path)

    model.eval()
    return model, tokenizer, resolved_model_path


def main():
    args = parse_args()
    transformers.set_seed(0)
    dtype = torch.bfloat16 if args.bf16 else torch.float16
    model, tokenizer, resolved_model_path = load_model_and_tokenizer(args.model_path, dtype, args.trust_remote_code)
    print(f"Model loaded from {resolved_model_path}")

    output_root = args.output_dir or args.model_path
    predictions_dir = os.path.join(output_root, "predictions")
    os.makedirs(predictions_dir, exist_ok=True)
    print(f"Predictions will be saved to {predictions_dir}")

    generation_config = build_generation_config(args, tokenizer)
    print(f"Generation config: {generation_config}")
    pipeline = TextGenerationPipeline(model=model, tokenizer=tokenizer)

    for benchmark in args.benchmarks:
        start_time = time.time()
        benchmark_path = os.path.join(args.data_path, benchmark)
        formatted = get_formatted_datasets(data_path=benchmark_path, prompt_only=True)

        responses: List[str] = []
        for response in tqdm(
            pipeline(
                KeyDataset(formatted["test"], "text"),
                generation_config=generation_config,
                **({"use_model_defaults": False} if supports_use_model_defaults() else {}),
                return_full_text=False,
                batch_size=args.batch_size,
            ),
            total=len(formatted["test"]),
        ):
            responses.append(response[0]["generated_text"])

        print(f"{benchmark} response example:\n{responses[0] if responses else ''}")
        elapsed = time.time() - start_time
        print(f"Time taken for {benchmark}: {elapsed:.2f} seconds")

        rows = []
        for i, row in enumerate(formatted["test"]):
            row["response"] = responses[i]
            rows.append(row)
        Dataset.from_list(rows).to_json(os.path.join(predictions_dir, f"{benchmark}_responses.jsonl"), lines=True)
        with open(os.path.join(predictions_dir, f"{benchmark}_time.txt"), "w", encoding="utf-8") as handle:
            handle.write(f"{elapsed:.2f}")


if __name__ == "__main__":
    main()
