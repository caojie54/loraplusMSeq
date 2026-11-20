"""Generate and score MedReason predictions with the SGLang offline engine."""

import argparse
import json
import os

from jinja2 import Template
from peft import PeftConfig
from tqdm import tqdm
from transformers import AutoTokenizer

from eval import load_file, postprocess_output
from scorer import get_results


DEFAULT_BENCHMARKS = [
    "medbullets_op4", "medbullets_op5", "HLE_biomed", "medqa_test", "MedXpertQA_test"
]


def resolve_model_path(model_path):
    merged_path = os.path.join(model_path, "merged")
    if os.path.isdir(merged_path):
        return merged_path
    adapter_path = os.path.join(model_path, "adapter")
    if os.path.isfile(os.path.join(adapter_path, "adapter_config.json")):
        return adapter_path
    return model_path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", "--model_path", default="meta-llama/Llama-2-7b-chat-hf",
                        help="Model ID, full/merged weights, LoRA adapter, or training output directory.")
    datasets = parser.add_mutually_exclusive_group()
    datasets.add_argument("--eval_file", help="Evaluate one JSON/JSONL file.")
    datasets.add_argument("--benchmarks", nargs="+", choices=DEFAULT_BENCHMARKS,
                          help="Benchmarks to evaluate; defaults to all five.")
    parser.add_argument("--data_path", default=os.path.abspath(os.path.join(os.path.dirname(__file__), "../eval_data")),
                        help="Directory containing <benchmark>.jsonl files.")
    parser.add_argument("--max_new_tokens", type=int, default=6000)
    parser.add_argument("--max_tokens", type=int, default=-1,
                        help="Keep at most this many prompt tokens from the beginning; <=0 disables truncation.")
    parser.add_argument("--use_chat_template", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--strict_prompt", action="store_true")
    parser.add_argument("--task", default="api", help="Filename suffix, matching eval.py.")
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--task_floder", default="anonymous_run")
    parser.add_argument("--results_dir", default="./results",
                        help="Result root; relative paths are resolved from the current working directory.")
    parser.add_argument("--tp_size", type=int, default=1)
    parser.add_argument("--dp_size", type=int, default=1)
    parser.add_argument("--mem_fraction_static", type=float, default=0.8)
    parser.add_argument("--context_length", type=int, default=None)
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--trust_remote_code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if min(args.batch_size, args.max_new_tokens, args.tp_size, args.dp_size) <= 0:
        parser.error("batch_size, max_new_tokens, tp_size and dp_size must be positive.")
    if not 0 < args.mem_fraction_static <= 1:
        parser.error("mem_fraction_static must be in (0, 1].")
    if args.context_length is not None and args.context_length <= 0:
        parser.error("context_length must be positive.")
    return args


def main():
    args = parse_args()
    eval_files = [args.eval_file] if args.eval_file else [
        os.path.join(args.data_path, name + ".jsonl")
        for name in (args.benchmarks or DEFAULT_BENCHMARKS)
    ]
    datasets = [(path, load_file(path)) for path in eval_files]
    resolved_model_path = resolve_model_path(args.model_name)
    tokenizer_path = resolved_model_path
    adapter_name = None
    lora_args = {}
    if os.path.isfile(os.path.join(resolved_model_path, "adapter_config.json")):
        config = PeftConfig.from_pretrained(resolved_model_path)
        if config.peft_type != "LORA":
            raise ValueError("SGLang evaluation requires LoRA adapters or merged model weights.")
        if not config.base_model_name_or_path:
            raise ValueError("The LoRA adapter config must specify base_model_name_or_path.")
        adapter_name = "evaluation_adapter"
        tokenizer_path = config.base_model_name_or_path
        lora_args = {
            "model_path": config.base_model_name_or_path,
            "enable_lora": True,
            "lora_paths": {adapter_name: resolved_model_path},
            "max_lora_rank": max([config.r, *config.rank_pattern.values()]),
        }
    tokenizer = None
    template = None
    if args.use_chat_template or args.max_tokens > 0:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, trust_remote_code=args.trust_remote_code, padding_side="left"
        )
    if args.use_chat_template:
        if not isinstance(tokenizer.chat_template, str) or not tokenizer.chat_template:
            raise ValueError("The tokenizer needs a string chat_template; use --no-use_chat_template for raw prompts.")
        template = Template(tokenizer.chat_template)

    if args.strict_prompt:
        query_prompt = "Please answer the following multiple-choice questions. Please answer the following multiple-choice questions, ensuring your response concludes with the correct option in the format: 'The answer is A.'.\n{question}\n{option_str}\n"
    else:
        query_prompt = "Please answer the following multiple-choice question:\n{question}\n{option_str}\n"

    engine_args = {
        "model_path": resolved_model_path,
        "dtype": args.dtype,
        "trust_remote_code": args.trust_remote_code,
        "tp_size": args.tp_size,
        "dp_size": args.dp_size,
        "mem_fraction_static": args.mem_fraction_static,
        "random_seed": args.seed,
    }
    engine_args.update(lora_args)
    if args.context_length is not None:
        engine_args["context_length"] = args.context_length
    sampling_params = {"temperature": 0.0, "top_p": 0.9, "max_new_tokens": args.max_new_tokens}
    engine = None
    try:
        if any(input_data for _, input_data in datasets):
            import sglang as sgl

            print(f"Loading SGLang offline engine from {resolved_model_path}", flush=True)
            engine = sgl.Engine(**engine_args)
            
        for eval_file, input_data in datasets:
            final_results = []
            for start in tqdm(range(0, len(input_data), args.batch_size), desc=os.path.basename(eval_file)):
                batch = input_data[start:start + args.batch_size]
                prompts = []
                for item in batch:
                    item["option_str"] = "\n".join(f"{op}. {answer}" for op, answer in item["options"].items())
                    item["input_str"] = query_prompt.format_map(item)
                    prompt = item["input_str"]
                    if template is not None:
                        prompt = template.render(
                            messages=[{"role": "user", "content": prompt}],
                            bos_token=tokenizer.bos_token,
                            add_generation_prompt=True,
                        )
                    if args.max_tokens > 0:
                        input_ids = tokenizer.encode(prompt, add_special_tokens=False)
                        if len(input_ids) > args.max_tokens:
                            prompt = tokenizer.decode(input_ids[:args.max_tokens])
                    prompts.append(prompt)
                if start == 0:
                    print(f"Example:\n{prompts[0]}")
                generate_args = {}
                if adapter_name is not None:
                    generate_args["lora_path"] = [adapter_name] * len(prompts)
                outputs = engine.generate(prompts, sampling_params, **generate_args)
                if len(outputs) != len(batch):
                    raise RuntimeError(f"Expected {len(batch)} outputs, got {len(outputs)}")
                for item, output in zip(batch, outputs):
                    finish_reason = output.get("meta_info", {}).get("finish_reason")
                    if isinstance(finish_reason, dict) and finish_reason.get("type") == "abort":
                        raise RuntimeError(f"Generation aborted: {finish_reason}")
                    prediction = postprocess_output(output["text"])
                    if prediction:
                        item["output"] = prediction
                        final_results.append(item)

            task_floder = os.path.join(args.results_dir, args.task_floder)
            os.makedirs(os.path.join(task_floder, "logs"), exist_ok=True)
            os.makedirs(os.path.join(task_floder, "result"), exist_ok=True)
            task_name = os.path.split(args.model_name)[-1]
            task_name += os.path.basename(eval_file).replace(".json", "") + f"_{args.task}"
            if args.strict_prompt:
                task_name += "_strict-prompt"
            log_save_path = os.path.join(task_floder, "logs", f"{task_name}.json")
            with open(log_save_path, "w", encoding="utf-8") as handle:
                json.dump(final_results, handle, ensure_ascii=False, indent=2)
            get_results(log_save_path)
    finally:
        if engine is not None:
            engine.shutdown()


if __name__ == "__main__":
    main()
