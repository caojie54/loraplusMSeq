"""Evaluate NaturalReasoning benchmark predictions."""

import argparse
import copy
import json
import os
import re


MC_BENCHMARKS = {
    "gpqa_diamond": set("ABCD"),
    "mmlu_pro": set("ABCDEFGHIJ"),
    "mmlu_pro_500": set("ABCDEFGHIJ"),
}


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def extract_letter(text, valid_letters):
    if not text:
        return ""
    upper = text.upper()
    patterns = [
        r"FINAL\s+ANSWER\s*[:：]?\s*\(?([A-J])\)?",
        r"ANSWER\s*[:：]\s*\(?([A-J])\)?",
        r"THE\s+ANSWER\s+IS\s*\(?([A-J])\)?",
        r"\(([A-J])\)",
        r"\b([A-J])\b",
    ]
    for pattern in patterns:
        for match in re.findall(pattern, upper):
            if match in valid_letters:
                return match
    return ""


def extract_boxed(text):
    marker = r"\boxed{"
    start = text.rfind(marker)
    if start == -1:
        return ""
    i = start + len(marker)
    depth = 1
    out = []
    while i < len(text) and depth:
        char = text[i]
        if char == "{":
            depth += 1
            out.append(char)
        elif char == "}":
            depth -= 1
            if depth:
                out.append(char)
        else:
            out.append(char)
        i += 1
    return "".join(out).strip()


def normalize_math(text):
    text = str(text or "")
    text = extract_boxed(text) or text.strip().splitlines()[-1] if text.strip() else ""
    replacements = {
        "\\left": "",
        "\\right": "",
        "\\!": "",
        "\\,": "",
        "$": "",
        " ": "",
        "\n": "",
        "\t": "",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"^answer[:：]?", "", text, flags=re.IGNORECASE)
    text = text.strip(".。")
    return text


def evaluate_multiple_choice(benchmark, rows):
    valid_letters = MC_BENCHMARKS[benchmark]
    checked = []
    correct = 0
    for row in rows:
        pred = extract_letter(row.get("response", ""), valid_letters)
        label = str(row.get("answer", "")).upper()
        flag = pred == label
        correct += int(flag)
        new_row = copy.deepcopy(row)
        new_row["pred"] = pred
        new_row["flag"] = flag
        checked.append(new_row)
    return correct / len(rows) if rows else 0.0, checked


def evaluate_math(rows):
    checked = []
    correct = 0
    for row in rows:
        pred = normalize_math(row.get("response", ""))
        label = normalize_math(row.get("answer", ""))
        flag = bool(pred) and pred == label
        correct += int(flag)
        new_row = copy.deepcopy(row)
        new_row["pred"] = pred
        new_row["normalized_answer"] = label
        new_row["flag"] = flag
        checked.append(new_row)
    return correct / len(rows) if rows else 0.0, checked


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate NaturalReasoning benchmark predictions.")
    parser.add_argument("--prediction_dir", type=str, required=True)
    parser.add_argument("--benchmarks", nargs="+", default=["gpqa_diamond", "math_500", "mmlu_pro_500"])
    return parser.parse_args()


def main():
    args = parse_args()
    results = {}
    for benchmark in args.benchmarks:
        path = os.path.join(args.prediction_dir, f"{benchmark}_responses.jsonl")
        rows = read_jsonl(path)
        if benchmark in MC_BENCHMARKS:
            acc, checked = evaluate_multiple_choice(benchmark, rows)
        elif benchmark == "math_500":
            acc, checked = evaluate_math(rows)
        else:
            raise ValueError(f"Unsupported benchmark: {benchmark}")
        results[benchmark] = acc
        write_jsonl(os.path.join(args.prediction_dir, f"{benchmark}_predict_checkanswer.jsonl"), checked)
        print(f"{benchmark}: accuracy {acc:.6f}")

    results["average"] = sum(results.values()) / len(results) if results else 0.0
    results["prediction_dir"] = args.prediction_dir
    score_path = os.path.join(os.path.dirname(args.prediction_dir), "natural_reasoning_acc_score.jsonl")
    with open(score_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(results, ensure_ascii=False) + "\n")
    print(f"acc:{results}")


if __name__ == "__main__":
    main()
