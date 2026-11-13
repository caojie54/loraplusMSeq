"""Plain-text MedReason SFT, with response-only loss and left truncation."""

from pathlib import Path

from datasets import load_dataset


def format_medreason(example):
    for key in ("question", "reasoning", "answer"):
        if not isinstance(example.get(key), str) or not example[key].strip():
            raise ValueError(f"MedReason requires a non-empty string field: {key}")
    # Match MedReason's field selection; do not append the separate options field.
    return {
        "prompt": example["question"] + "\n\n",
        "response": "## Thinking\n\n{}\n\n## Final Answer\n\n{}".format(
            example["reasoning"], example["answer"]
        ),
    }


def load_medreason(data_path):
    path = Path(data_path)
    if path.is_dir():
        path = path / "ours_quality_33000.jsonl"
    if not path.is_file():
        raise FileNotFoundError(path)
    return load_dataset("json", data_files={"train": str(path)}).map(format_medreason)


def tokenize_medreason(example, tokenizer, max_length):
    if max_length < 2 or tokenizer.eos_token_id is None:
        raise ValueError("MedReason requires max_length >= 2 and an EOS token")
    prompt_ids = tokenizer.encode(example["prompt"], add_special_tokens=False)
    response_ids = tokenizer.encode(example["response"], add_special_tokens=False)
    if not response_ids or response_ids[-1] != tokenizer.eos_token_id:
        response_ids.append(tokenizer.eos_token_id)
    ids = prompt_ids + response_ids
    labels = [-100] * len(prompt_ids) + response_ids.copy()
    # Same truncation direction as MedReason/SFT.py: preserve the final answer.
    return {
        "input_ids": ids[-max_length:],
        "attention_mask": [1] * min(len(ids), max_length),
        "labels": labels[-max_length:],
    }
