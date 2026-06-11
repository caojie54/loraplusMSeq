"""Generate commonsense benchmark predictions for a trained model."""

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


COMMONSENSE_DATASETS = [
    "boolq",
    "piqa",
    "social_i_qa",
    "hellaswag",
    "winogrande",
    "ARC-Challenge",
    "ARC-Easy",
    "openbookqa",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run commonsense generation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="/mnt/petrelfs/caojie1/projects/CoMoL/datasets/math_commonsense")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--trust_remote_code", action="store_true", default=True)
    return parser.parse_args()


def configure_tokenizer(tokenizer, model_path: str) -> None:
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is not None:
        return
    if "llama-3" in model_path.lower():
        tokenizer.pad_token_id = 128255
    else:
        tokenizer.pad_token = tokenizer.eos_token


def resolve_model_path(model_path: str) -> str:
    if os.path.isdir(os.path.join(model_path, "merged")):
        return os.path.join(model_path, "merged")
    adapter_path = os.path.join(model_path, "adapter")
    if os.path.exists(os.path.join(adapter_path, "adapter_config.json")):
        return adapter_path
    return model_path


def load_model_and_tokenizer(model_path: str, dtype: torch.dtype, trust_remote_code: bool):
    model_path = resolve_model_path(model_path)
    if os.path.exists(os.path.join(model_path, "adapter_config.json")):
        peft_config = PeftConfig.from_pretrained(model_path)
        base_model = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=trust_remote_code,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            peft_config.base_model_name_or_path,
            padding_side="left",
            trust_remote_code=trust_remote_code,
        )
        model = PeftModel.from_pretrained(base_model, model_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=trust_remote_code,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left", trust_remote_code=trust_remote_code)

    configure_tokenizer(tokenizer, model_path)
    model.eval()
    return model, tokenizer, model_path


def main():
    args = parse_args()
    transformers.set_seed(0)
    dtype = torch.bfloat16 if args.bf16 else torch.float16
    model, tokenizer, resolved_model_path = load_model_and_tokenizer(args.model_path, dtype, args.trust_remote_code)
    print(f"Model loaded from {resolved_model_path}")

    prediction_dir = os.path.join(args.model_path, "predictions")
    os.makedirs(prediction_dir, exist_ok=True)
    generation_config = GenerationConfig(max_new_tokens=args.max_new_tokens, do_sample=False)
    pipeline = TextGenerationPipeline(model=model, tokenizer=tokenizer)

    for data_name in COMMONSENSE_DATASETS:
        start_time = time.time()
        data_path_1 = os.path.join(args.data_path, data_name)
        formatted_datasets = get_formatted_datasets(data_path=data_path_1, prompt_only=True)

        responses: List[str] = []
        for response in tqdm(
            pipeline(
                KeyDataset(formatted_datasets["test"], "text"),
                generation_config=generation_config,
                return_full_text=False,
                batch_size=args.batch_size,
            ),
            total=len(formatted_datasets["test"]),
        ):
            responses.append(response[0]["generated_text"])

        print(f"Response example for {data_name}:\n{responses[0]}")
        end_time = time.time()
        print(f"Time taken for {data_name}: {end_time - start_time:.2f} seconds")

        new_data = []
        for i, x in enumerate(formatted_datasets["test"]):
            x["response"] = responses[i]
            new_data.append(x)

        output_name = data_name.lower()
        Dataset.from_list(new_data).to_json(os.path.join(prediction_dir, f"{output_name}_responses.jsonl"), lines=True)
        with open(os.path.join(prediction_dir, f"{output_name}_time.txt"), "w", encoding="utf-8") as f:
            f.write(f"{end_time - start_time:.2f}")


if __name__ == "__main__":
    main()
