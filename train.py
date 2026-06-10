"""Sequential LoRA plus compensation-module fine-tuning."""

import argparse
import json
import os
import re
from typing import Dict

import torch
import transformers
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling

from loraplusmseq.data import get_formatted_datasets
from loraplusmseq.module_selection import CompensationModuleManager, set_lora_only_trainable
from loraplusmseq.seq_trainer import SequentialLoraPlusMTrainer


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def sanitize_name(text: str) -> str:
    return re.sub(r"[^0-9a-zA-Z._-]+", "-", text).strip("-")


def build_output_dir(args) -> str:
    if args.run_name:
        return os.path.join(args.output_dir, sanitize_name(args.run_name))

    model_name = os.path.basename(args.model_path).lower()
    data_name = os.path.basename(args.data_path).lower()
    targets = "".join([x.split("_")[0] for x in args.target_modules])
    method_name = f"seq-{args.method}"
    if args.method != "lora":
        if args.compensation_top_k > 0:
            method_name += f"-top{args.compensation_top_k}"
        else:
            method_name += f"-ratio{args.compensation_ratio}"
        method_name += f"-interval{args.selection_interval}"
        if args.method == "alpha":
            method_name += f"-{args.alpha_score}"
    name = f"{model_name}-{method_name}-{targets}-rank{args.lora_rank}-{data_name}-epoch{args.num_train_epochs:g}"
    if args.seed != 0:
        name += f"-seed{args.seed}"
    return os.path.join(args.output_dir, sanitize_name(name))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sequential LoRA then compensation-module fine-tuning.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model_path", type=str, default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--data_path", type=str, default="/mnt/petrelfs/caojie1/projects/CoMoL/datasets/commonsense170k")
    parser.add_argument("--output_dir", type=str, default="/mnt/dhwfile/raise/user/caojie/loraplusMSeq/outputs/commonsense170k")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument(
        "--method",
        type=str,
        default="alpha",
        choices=["lora", "static_random", "dynamic_random", "alpha"],
        help="Sequential method/baseline. lora skips module replay.",
    )
    parser.add_argument("--target_modules", type=str, nargs="+", default=["q_proj", "v_proj"])
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=None)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="constant_with_warmup",
        choices=["constant", "constant_with_warmup", "linear"],
    )
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--selection_interval", type=int, default=50, help="Number of LoRA optimizer steps per n-batch.")
    parser.add_argument("--compensation_top_k", type=int, default=8, help="Number of original modules to train.")
    parser.add_argument(
        "--compensation_ratio",
        type=float,
        default=0.0,
        help="Candidate-parameter budget used only when compensation_top_k <= 0.",
    )
    parser.add_argument(
        "--alpha_score",
        type=str,
        default="lora_update_ratio",
        choices=["lora_update_ratio", "lora_grad_norm"],
        help="Score accumulated from LoRA gradients during the LoRA phase.",
    )
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--max_eval_samples", type=int, default=0)
    parser.add_argument("--bf16", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--gradient_checkpointing", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--save_merged_model", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--trust_remote_code", type=str2bool, nargs="?", const=True, default=True)
    return parser.parse_args()


def configure_tokenizer(tokenizer, model_path: str) -> None:
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is not None:
        return
    if "llama-3" in model_path.lower():
        tokenizer.pad_token_id = 128255
    else:
        tokenizer.pad_token = tokenizer.eos_token


def main():
    args = parse_args()
    transformers.set_seed(args.seed)

    output_dir = build_output_dir(args)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Arguments: {args}")
    print(f"Resolved output_dir: {output_dir}")

    formatted_datasets = get_formatted_datasets(data_path=args.data_path, prompt_only=False)
    if args.max_train_samples > 0:
        formatted_datasets["train"] = formatted_datasets["train"].select(
            range(min(args.max_train_samples, len(formatted_datasets["train"])))
        )
    if "validation" in formatted_datasets and args.max_eval_samples > 0:
        formatted_datasets["validation"] = formatted_datasets["validation"].select(
            range(min(args.max_eval_samples, len(formatted_datasets["validation"])))
        )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        padding_side="left",
        trust_remote_code=args.trust_remote_code,
    )
    configure_tokenizer(tokenizer, args.model_path)

    def tokenize_text(examples: Dict[str, str]) -> Dict[str, object]:
        out = tokenizer(examples["text"], truncation=True, max_length=args.max_length)
        if out["input_ids"] and out["input_ids"][-1] != tokenizer.eos_token_id and len(out["input_ids"]) < args.max_length:
            out["input_ids"].append(tokenizer.eos_token_id)
            out["attention_mask"].append(1)
        return out

    tokenized_datasets = formatted_datasets.map(
        tokenize_text,
        batched=False,
        remove_columns=formatted_datasets["train"].column_names,
    )
    print(f"Tokenized datasets: {tokenized_datasets}")

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False, pad_to_multiple_of=8, return_tensors="pt")

    dtype = torch.bfloat16 if args.bf16 else torch.float16
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=args.trust_remote_code,
    )
    base_model.config.use_cache = False
    if args.gradient_checkpointing:
        base_model.gradient_checkpointing_enable()

    lora_alpha = args.lora_alpha if args.lora_alpha is not None else args.lora_rank
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=args.lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
        bias="none",
    )
    model = get_peft_model(base_model, peft_config)
    model.enable_input_require_grads()
    set_lora_only_trainable(model)
    model.print_trainable_parameters()

    history_path = os.path.join(output_dir, "selection_history.jsonl")
    if os.path.exists(history_path):
        os.remove(history_path)
    module_manager = CompensationModuleManager(
        model=model,
        target_modules=args.target_modules,
        method=args.method,
        top_k=args.compensation_top_k,
        param_ratio=args.compensation_ratio,
        seed=args.seed,
        alpha_score=args.alpha_score,
        history_path=history_path,
    )
    module_manager.save_candidates(os.path.join(output_dir, "candidate_modules.json"))

    with open(os.path.join(output_dir, "loraplusmseq_args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    trainer = SequentialLoraPlusMTrainer(
        model=model,
        train_dataset=tokenized_datasets["train"],
        data_collator=data_collator,
        module_manager=module_manager,
        output_dir=output_dir,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        selection_interval=args.selection_interval,
        logging_steps=args.logging_steps,
        seed=args.seed,
        bf16=args.bf16,
        max_grad_norm=args.max_grad_norm,
        dataloader_num_workers=args.dataloader_num_workers,
    )
    trainer.train()

    adapter_dir = os.path.join(output_dir, "adapter")
    model.save_pretrained(adapter_dir, safe_serialization=True)
    tokenizer.save_pretrained(adapter_dir)
    print(f"Adapter saved to {adapter_dir}")

    if args.save_merged_model:
        merged_dir = os.path.join(output_dir, "merged")
        merged_model = model.merge_and_unload()
        merged_model.config.use_cache = True
        merged_model.save_pretrained(merged_dir, safe_serialization=True)
        tokenizer.save_pretrained(merged_dir)
        print(f"Merged model saved to {merged_dir}")


if __name__ == "__main__":
    main()

