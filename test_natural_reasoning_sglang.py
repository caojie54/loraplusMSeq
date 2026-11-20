"""Generate NaturalReasoning predictions with the SGLang offline engine."""

import argparse
import os
import time

from datasets import Dataset
from peft import PeftConfig
from tqdm import tqdm

from loraplusmseq.data import get_formatted_datasets

DEFAULT_DO_SAMPLE = False
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_P = 0.95


def str2bool(value):
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"true", "1", "yes", "y", "on"}:
        return True
    if lowered in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def resolve_model_path(model_path: str) -> str:
    merged_path = os.path.join(model_path, "merged")
    if os.path.isdir(merged_path):
        return merged_path
    adapter_path = os.path.join(model_path, "adapter")
    if os.path.exists(os.path.join(adapter_path, "adapter_config.json")):
        return adapter_path
    return model_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate GPQA-Diamond, MATH-500, and MMLU-Pro-500 predictions with SGLang.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_path", default="/data/user/cj/projects/loraplusMSeq/datasets/natural_reasoning_eval")
    parser.add_argument("--benchmarks", nargs="+", default=["gpqa_diamond", "math_500", "mmlu_pro_500"])
    parser.add_argument("--output_dir", default=None, help="Prediction root; defaults to model_path.")

    parser.add_argument("--max_new_tokens", type=int, default=1536)
    parser.add_argument("--batch_size", type=int, default=64)
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
    
    parser.add_argument("--bf16", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--trust_remote_code", type=str2bool, nargs="?", const=True, default=True)

    parser.add_argument("--tp_size", type=int, default=1)
    parser.add_argument("--dp_size", type=int, default=1)
    parser.add_argument("--mem_fraction_static", type=float, default=0.8)
    parser.add_argument("--context_length", type=int, default=None, help="Override the engine context length.")
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    if min(args.batch_size, args.max_new_tokens, args.tp_size, args.dp_size) <= 0:
        parser.error("Batch size, token limit, tp_size and dp_size must be positive.")

    if not 0 < args.mem_fraction_static <= 1:
        parser.error("mem_fraction_static must be in (0, 1].")

    if args.context_length is not None and args.context_length <= 0:
        parser.error("context_length must be positive.")

    if args.do_sample and args.temperature is not None and args.temperature <= 0:
        parser.error("Sampling requires a positive temperature.")

    if args.top_p is not None and not 0 < args.top_p <= 1:
        parser.error("top_p must be in (0, 1].")

    return args


def main():
    args = parse_args()
    import sglang as sgl

    resolved_model_path = resolve_model_path(args.model_path)
    engine_args = {
        "model_path": resolved_model_path,
        "dtype": "bfloat16" if args.bf16 else "float16",
        "trust_remote_code": args.trust_remote_code,
        "tp_size": args.tp_size,
        "dp_size": args.dp_size,
        "mem_fraction_static": args.mem_fraction_static,
        "random_seed": args.seed,
    }
    if args.context_length is not None:
        engine_args["context_length"] = args.context_length

    # lora supported
    adapter_name = None
    if os.path.isfile(os.path.join(resolved_model_path, "adapter_config.json")):
        config = PeftConfig.from_pretrained(resolved_model_path)
        if config.peft_type != "LORA":
            raise ValueError("SGLang evaluation requires LoRA adapters or merged model weights.")
        adapter_name = "evaluation_adapter"
        engine_args.update(
            model_path=config.base_model_name_or_path,
            enable_lora=True,
            lora_paths={adapter_name: resolved_model_path},
            max_lora_rank=max([config.r, *config.rank_pattern.values()]),
        )

    # Match the original script's greedy default and GenerationConfig top-k=50.
    sampling_params = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 1,
        "repetition_penalty": 1.0,
    }
    if args.do_sample:
        sampling_params.update(
            temperature=DEFAULT_TEMPERATURE if args.temperature is None else args.temperature,
            top_p=DEFAULT_TOP_P if args.top_p is None else args.top_p,
            top_k=64,
        )

    predictions_dir = os.path.join(args.output_dir or args.model_path, "predictions")
    os.makedirs(predictions_dir, exist_ok=True)
    print(f"Loading model from {resolved_model_path}", flush=True)
    print(f"Predictions will be saved to {predictions_dir}", flush=True)
    print(f"Sampling parameters: {sampling_params}", flush=True)
    engine = sgl.Engine(**engine_args)
    try:
        for benchmark in args.benchmarks:
            start_time = time.perf_counter()
            formatted = get_formatted_datasets(
                data_path=os.path.join(args.data_path, benchmark), prompt_only=True
            )
            dataset = formatted["test"]
            responses = []
            for start in tqdm(range(0, len(dataset), args.batch_size), desc=benchmark):
                prompts = dataset[start:start + args.batch_size]["text"]
                generate_args = {}
                if adapter_name is not None:
                    generate_args["lora_path"] = [adapter_name] * len(prompts)
                outputs = engine.generate(prompts, sampling_params, **generate_args)
                if len(outputs) != len(prompts):
                    raise RuntimeError(f"{benchmark}: expected {len(prompts)} outputs, got {len(outputs)}")
                for output in outputs:
                    finish_reason = output.get("meta_info", {}).get("finish_reason")
                    if isinstance(finish_reason, dict) and finish_reason.get("type") == "abort":
                        raise RuntimeError(f"{benchmark}: generation aborted: {finish_reason}")
                    responses.append(output["text"])

            elapsed = time.perf_counter() - start_time
            print(f"{benchmark} response example:\n{responses[0] if responses else ''}")
            print(f"Time taken for {benchmark}: {elapsed:.2f} seconds")
            rows = [dict(row, response=response) for row, response in zip(dataset, responses)]
            prediction_path = os.path.join(predictions_dir, f"{benchmark}_responses.jsonl")
            Dataset.from_list(rows).to_json(prediction_path, lines=True)
            
            with open(os.path.join(predictions_dir, f"{benchmark}_time.txt"), "w", encoding="utf-8") as handle:
                handle.write(f"{elapsed:.2f}")
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()
