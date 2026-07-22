"""Output-only fine-tuning for MedMCQA and synthetic text-to-SQL."""

import argparse
import json
import os
import re
from typing import Dict

import torch
import transformers
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq

from generation_test_utils import get_formatted_datasets
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
        method_name += f"-ratio{args.compensation_ratio}"
        method_name += f"-interval{args.selection_interval}"
        if args.method == "alpha":
            method_name += f"-{args.alpha_score}"
            if args.alpha_candidate_ratio > args.compensation_ratio:
                method_name += f"-candidate_ratio{args.alpha_candidate_ratio}"
        if args.lora_optimizer_reset_strategy != "keep":
            method_name += f"-loraopt{args.lora_optimizer_reset_strategy}"
        if args.module_optimizer_state_strategy != "reset_offload":
            method_name += f"-moduleopt{args.module_optimizer_state_strategy}"
        if args.lora_optimizer_dtype != "bf16":
            method_name += f"-loraopt{args.lora_optimizer_dtype}"
        if args.module_optimizer_dtype != "bf16":
            method_name += f"-moduleoptdtype{args.module_optimizer_dtype}"
        if args.module_gradient_mode != "full":
            method_name += f"-modulegrad{args.module_gradient_mode}"
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
    parser.add_argument("--lora_dropout", type=float, default=0)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--module_learning_rate", type=float, default=1e-5)
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
    parser.add_argument(
        "--compensation_ratio",
        type=float,
        default=0.005,
        help="Fraction of original model parameters to train in each module replay block.",
    )
    parser.add_argument(
        "--alpha_score",
        type=str,
        default="lora_grad_norm",
        choices=[
            "lora_update_ratio",
            "lora_grad_norm",
            "lora_grad_norm_min",
            "lora_effective_update_pressure",
        ],
        help="Score accumulated from LoRA gradients during the LoRA phase.",
    )
    parser.add_argument(
        "--alpha_candidate_ratio",
        type=float,
        default=0.0,
        help="Larger ratio2 candidate-pool budget for alpha sampling. Disabled unless greater than compensation_ratio.",
    )
    parser.add_argument(
        "--alpha_sampling_temperature",
        type=float,
        default=1.0,
        help="Softmax temperature for alpha ratio2 importance sampling.",
    )
    parser.add_argument(
        "--alpha_uniform_mix",
        type=float,
        default=0.1,
        help="Uniform exploration mixture for alpha ratio2 importance sampling.",
    )
    parser.add_argument(
        "--alpha_score_gamma",
        type=float,
        default=1.0,
        help="Parameter-count penalty on alpha sampling scores: log(score) - gamma * log(num_params).",
    )
    parser.add_argument(
        "--alpha_group_norm",
        type=str,
        default="none",
        choices=["none", "global", "type"],
        help="Optional z-score normalization for alpha sampling scores before building the ratio2 pool.",
    )
    parser.add_argument(
        "--lora_optimizer_reset_strategy",
        type=str,
        default="keep",
        choices=["keep", "reset_all", "reset_selected"],
        help="How to reset LoRA optimizer state after each module replay block.",
    )
    parser.add_argument(
        "--module_optimizer_state_strategy",
        type=str,
        default="reset_offload",
        choices=["reset_offload", "persistent_offload"],
        help="How to handle module AdamW state across module replay blocks. reset_offload recreates the module optimizer each block and frees/offloads it before returning to LoRA.",
    )
    parser.add_argument(
        "--lora_optimizer_dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp32"],
        help="Use normal low-precision AdamW or fp32 shadow AdamW for LoRA updates.",
    )
    parser.add_argument(
        "--module_optimizer_dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp32"],
        help="Use normal low-precision AdamW or fp32 shadow AdamW for module replay updates.",
    )
    parser.add_argument(
        "--module_gradient_mode",
        type=str,
        default="full",
        choices=["full", "residual"],
        help="Use full selected-module gradients or project them to the LoRA residual subspace.",
    )
    parser.add_argument("--residual_rtol", type=float, default=1e-4)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--train_sample_offset", type=int, default=0)
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


def tokenize_supervised_example(example: Dict[str, str], tokenizer, max_length: int) -> Dict[str, object]:
    """Tokenize a prompt/target pair while supervising only target tokens."""
    if max_length <= 0:
        raise ValueError(f"max_length must be positive, got {max_length}")
    if tokenizer.eos_token_id is None:
        raise ValueError("The tokenizer must define eos_token_id for supervised training.")

    prompt = str(example["text"])
    target = str(example.get("output") or "")
    if not target:
        raise ValueError("Training examples must contain a non-empty output target.")

    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(target, add_special_tokens=False)["input_ids"]
    if not target_ids or target_ids[-1] != tokenizer.eos_token_id:
        target_ids.append(tokenizer.eos_token_id)

    # Preserve the target and its EOS. Only the prompt gives up space first.
    if len(target_ids) > max_length:
        target_ids = target_ids[:max_length]
        target_ids[-1] = tokenizer.eos_token_id
    prompt_budget = max_length - len(target_ids)
    if len(prompt_ids) > prompt_budget:
        if tokenizer.truncation_side == "left":
            prompt_ids = prompt_ids[-prompt_budget:] if prompt_budget else []
        else:
            prompt_ids = prompt_ids[:prompt_budget]

    input_ids = prompt_ids + target_ids
    labels = [-100] * len(prompt_ids) + target_ids.copy()
    assert len(input_ids) == len(labels) <= max_length
    assert labels[-len(target_ids) :] == target_ids
    assert any(label != -100 for label in labels)
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }



def count_original_llm_params(model: torch.nn.Module) -> int:
    total = int(sum(param.numel() for name, param in model.named_parameters() if "lora_" not in name))
    if total > 0:
        return total
    return int(sum(param.numel() for param in model.parameters()))


def resolve_peft_target_modules(model: torch.nn.Module, target_modules):
    """Limit multimodal Gemma-4 targets to the language model path."""
    target_set = set(target_modules)
    language_targets = [
        name
        for name, module in model.named_modules()
        if name.startswith("model.language_model.")
        and name.split(".")[-1] in target_set
        and isinstance(module, torch.nn.Linear)
    ]
    if language_targets:
        print(
            f"Resolved PEFT target modules to {len(language_targets)} language_model modules "
            f"from suffixes={list(target_modules)}",
            flush=True,
        )
        return language_targets
    return list(target_modules)


def write_training_success_marker(
    output_dir: str,
    metrics: Dict[str, object],
) -> str:
    path = os.path.join(output_dir, "TRAIN_SUCCESS")
    temporary = f"{path}.{os.getpid()}.tmp"
    payload = {
        "train_updates": metrics.get("train_updates"),
        "trainer": "loraplusmseq.seq_trainer.SequentialLoraPlusMTrainer",
    }
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return path


def main():
    args = parse_args()
    transformers.set_seed(args.seed)

    output_dir = build_output_dir(args)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Arguments: {args}")
    print(f"Resolved output_dir: {output_dir}")

    formatted_datasets = get_formatted_datasets(data_path=args.data_path, prompt_only=True)
    train_dataset = formatted_datasets["train"]
    if args.max_train_samples > 0:
        start = min(max(args.train_sample_offset, 0), len(train_dataset))
        end = min(start + args.max_train_samples, len(train_dataset))
        if start == end:
            raise ValueError(f"Empty train sample range: offset={args.train_sample_offset}, size={args.max_train_samples}")
        train_dataset = train_dataset.select(
            range(start, end)
        )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        padding_side="left",
        trust_remote_code=args.trust_remote_code,
    )
    configure_tokenizer(tokenizer, args.model_path)

    tokenized_train_dataset = train_dataset.map(
        lambda example: tokenize_supervised_example(example, tokenizer, args.max_length),
        batched=False,
        remove_columns=train_dataset.column_names,
        load_from_cache_file=False,
        keep_in_memory=True,
    )
    print(f"Tokenized train dataset: {tokenized_train_dataset}")

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        pad_to_multiple_of=8,
        label_pad_token_id=-100,
        return_tensors="pt",
    )

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

    peft_target_modules = resolve_peft_target_modules(base_model, args.target_modules)
    lora_alpha = args.lora_alpha if args.lora_alpha is not None else args.lora_rank
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=args.lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=peft_target_modules,
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
        target_modules=peft_target_modules,
        method=args.method,
        param_ratio=args.compensation_ratio,
        total_model_params=count_original_llm_params(model),
        seed=args.seed,
        alpha_score=args.alpha_score,
        alpha_candidate_ratio=args.alpha_candidate_ratio,
        alpha_sampling_temperature=args.alpha_sampling_temperature,
        alpha_uniform_mix=args.alpha_uniform_mix,
        alpha_score_gamma=args.alpha_score_gamma,
        alpha_group_norm=args.alpha_group_norm,
        history_path=history_path,
    )
    module_manager.save_candidates(os.path.join(output_dir, "candidate_modules.json"))

    with open(os.path.join(output_dir, "loraplusmseq_args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    trainer = SequentialLoraPlusMTrainer(
        model=model,
        train_dataset=tokenized_train_dataset,
        data_collator=data_collator,
        module_manager=module_manager,
        output_dir=output_dir,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        module_learning_rate=args.module_learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        selection_interval=args.selection_interval,
        logging_steps=args.logging_steps,
        seed=args.seed,
        bf16=args.bf16,
        max_grad_norm=args.max_grad_norm,
        dataloader_num_workers=args.dataloader_num_workers,
        lora_optimizer_reset_strategy=args.lora_optimizer_reset_strategy,
        module_optimizer_state_strategy=args.module_optimizer_state_strategy,
        lora_optimizer_dtype=args.lora_optimizer_dtype,
        module_optimizer_dtype=args.module_optimizer_dtype,
        module_gradient_mode=args.module_gradient_mode,
        residual_rtol=args.residual_rtol,
    )
    train_metrics = trainer.train()

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

    success_marker = write_training_success_marker(output_dir, train_metrics)
    print(f"Training success marker written to {success_marker}", flush=True)


if __name__ == "__main__":
    main()
