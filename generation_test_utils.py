"""Shared dataset formatting and batched generation utilities."""

import argparse
import json
import os
import time
from typing import Any, Dict

import torch
import transformers
from datasets import load_dataset
from peft import PeftConfig, PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, TextGenerationPipeline
from transformers.pipelines.pt_utils import KeyDataset


DEFAULT_DO_SAMPLE = False
DEFAULT_TEMPERATURE = 0.6
DEFAULT_TOP_P = 0.9

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}


def format_text(example: Dict[str, Any], data_name: str, prompt_only: bool = True) -> Dict[str, Any]:
    """Format one instruction example into the prompt style used by CoMoL."""
    if data_name == "humaneval":
        example["instruction"] = example["prompt"]
    elif data_name == "mbpp":
        instruction = example["text"]
        instruction = instruction + "\nExamples:\n" + "\n".join(example["test_list"])
        example["instruction"] = instruction

    if example.get("prompt"):
        text = example["prompt"]
    elif "input" in example and example["input"]:
        text = PROMPT_DICT["prompt_input"].format_map(example)
    else:
        text = PROMPT_DICT["prompt_no_input"].format_map(example)

    if not prompt_only:
        text += f"{example['output']}"

    example["data_name"] = data_name
    example["text"] = text
    return example


def get_formatted_datasets(data_path: str, prompt_only: bool):
    """Load a HF/local dataset and add the formatted ``text`` column."""
    data_name = os.path.basename(data_path).lower()
    datasets = load_dataset(path=data_path)
    first_split = next(iter(datasets))
    print(f"Datasets: {datasets}")
    print(f"Example: {datasets[first_split][0]}")

    formatted_datasets = datasets.map(
        lambda example: format_text(example, data_name, prompt_only=prompt_only),
        batched=False,
        load_from_cache_file=False,
        keep_in_memory=True,
    )
    print(f"Formatted datasets: {formatted_datasets}")
    print(f"Formatted example: {formatted_datasets[first_split][0]}")
    print(f"Text example:\n{formatted_datasets[first_split]['text'][0]}")
    return formatted_datasets


def str2bool(value):
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"true", "1", "yes", "y", "on"}:
        return True
    if lowered in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def add_generation_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--do_sample",
        type=str2bool,
        nargs="?",
        const=True,
        default=None,
        help="Use sampling. Defaults to false/greedy when omitted; pass --do_sample true for sampling.",
    )
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_p", type=float, default=None)


def parse_generation_args(description: str):
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_samples", type=int, default=0, help="Limit generation for smoke/memory tests; 0 uses all rows.")
    add_generation_args(parser)
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Optional root directory for predictions; defaults to model_path.",
    )
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trust_remote_code", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def supports_use_model_defaults() -> bool:
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


def run_generation(args, benchmark: str) -> None:
    """Generate predictions for one fixed benchmark."""
    transformers.set_seed(0)
    dtype = torch.bfloat16 if args.bf16 else torch.float16
    model, tokenizer, resolved_model_path = load_model_and_tokenizer(args.model_path, dtype, args.trust_remote_code)
    print(f"Model loaded from {resolved_model_path}")

    output_root = args.output_dir or args.model_path
    prediction_dir = os.path.join(output_root, "predictions")
    os.makedirs(prediction_dir, exist_ok=True)
    print(f"Predictions will be saved to {prediction_dir}")

    generation_config = build_generation_config(args, tokenizer)
    print(f"Generation config: {generation_config}")
    pipeline = TextGenerationPipeline(model=model, tokenizer=tokenizer)

    start_time = time.time()
    benchmark_path = os.path.join(args.data_path, benchmark)
    formatted = get_formatted_datasets(data_path=benchmark_path, prompt_only=True)
    test_dataset = formatted["test"]
    if args.max_samples > 0:
        test_dataset = test_dataset.select(range(min(args.max_samples, len(test_dataset))))

    response_iter = pipeline(
        KeyDataset(test_dataset, "text"),
        generation_config=generation_config,
        **({"use_model_defaults": False} if supports_use_model_defaults() else {}),
        return_full_text=False,
        batch_size=args.batch_size,
    )
    response_path = os.path.join(prediction_dir, f"{benchmark}_responses.jsonl")
    example_response = ""
    with open(response_path, "w", encoding="utf-8") as handle:
        for row, response in zip(test_dataset, tqdm(response_iter, total=len(test_dataset))):
            generated = response[0]["generated_text"]
            if not example_response:
                example_response = generated
            output_row = dict(row)
            output_row["response"] = generated
            handle.write(json.dumps(output_row, ensure_ascii=False) + "\n")

    print(f"{benchmark} response example:\n{example_response}")
    elapsed = time.time() - start_time
    print(f"Time taken for {benchmark}: {elapsed:.2f} seconds")
    with open(os.path.join(prediction_dir, f"{benchmark}_time.txt"), "w", encoding="utf-8") as handle:
        handle.write(f"{elapsed:.2f}")
