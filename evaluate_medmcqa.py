"""Evaluate MedMCQA predictions."""

import argparse
import copy
import json
import os
import re
from typing import Any, Dict, List, Tuple


def read_jsonl(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: str, rows: List[Dict]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def extract_letter(text: str) -> str:
    """Extract an explicit MedMCQA option without scanning arbitrary letters."""
    if not text:
        return ""
    text = str(text).strip()
    explicit_patterns = (
        r"\bFINAL\s+ANSWER\s*(?:IS\s*)?[:：]?\s*\(?([A-D])\b\)?",
        r"\bTHE\s+(?:CORRECT\s+)?ANSWER\s+IS\s*\(?([A-D])\b\)?",
        r"\bANSWER\s*(?:IS\s*|[:：]\s*)\(?([A-D])\b\)?",
        r"\b(?:OPTION|CHOICE)\s*(?:IS\s*)?[:：]?\s*\(?([A-D])\b\)?",
    )
    explicit_matches = []
    for pattern in explicit_patterns:
        explicit_matches.extend(re.finditer(pattern, text, flags=re.IGNORECASE))
    if explicit_matches:
        return max(explicit_matches, key=lambda match: match.start()).group(1).upper()

    # A bare leading option must be complete or followed by punctuation/newline.
    match = re.match(
        r"^\s*(?:\*\*)?\(?([A-D])\)?(?:\*\*)?(?=\s*(?:$|[\n\r\).,:：-]))",
        text,
        flags=re.IGNORECASE,
    )
    return match.group(1).upper() if match else ""


def medmcqa_label(row: Dict) -> str:
    label = extract_letter(row.get("answer", "")) or extract_letter(row.get("output", ""))
    if label:
        return label
    for key in ("answer_index", "cop"):
        value = row.get(key)
        if isinstance(value, str) and value.strip().isdigit():
            value = int(value.strip())
        if isinstance(value, int) and not isinstance(value, bool) and 0 <= value < 4:
            return "ABCD"[value]
    return ""


def evaluate_medmcqa_metrics(rows: List[Dict]) -> Tuple[Dict[str, Any], List[Dict]]:
    checked = []
    correct = 0
    valid = 0
    for row in rows:
        pred = extract_letter(row.get("response", ""))
        label = medmcqa_label(row)
        prediction_valid = bool(pred)
        flag = prediction_valid and bool(label) and pred == label
        valid += int(prediction_valid)
        correct += int(flag)
        new_row = copy.deepcopy(row)
        new_row["pred"] = pred
        new_row["normalized_answer"] = label
        new_row["prediction_valid"] = prediction_valid
        new_row["flag"] = flag
        checked.append(new_row)

    total = len(rows)
    metrics = {
        "accuracy": correct / total if total else 0.0,
        "correct": correct,
        "total": total,
        "valid_prediction_rate": valid / total if total else 0.0,
        "invalid_count": total - valid,
    }
    return metrics, checked


def evaluate_medmcqa(rows: List[Dict]) -> Tuple[float, List[Dict]]:
    """Compatibility wrapper returning only the primary score and rows."""
    metrics, checked = evaluate_medmcqa_metrics(rows)
    return metrics["accuracy"], checked


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MedMCQA predictions.")
    parser.add_argument("--prediction_dir", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    benchmark = "medmcqa"
    path = os.path.join(args.prediction_dir, f"{benchmark}_responses.jsonl")
    rows = read_jsonl(path)
    metrics, checked = evaluate_medmcqa_metrics(rows)
    primary = metrics["accuracy"]
    results: Dict[str, Any] = {
        benchmark: primary,
        f"{benchmark}_metrics": metrics,
        "average": primary,
        "prediction_dir": args.prediction_dir,
    }
    write_jsonl(
        os.path.join(args.prediction_dir, f"{benchmark}_predict_checkanswer.jsonl"),
        checked,
    )
    print(f"{benchmark}: accuracy {primary:.6f}")

    score_path = os.path.join(
        os.path.dirname(args.prediction_dir),
        "medmcqa_scores.jsonl",
    )
    with open(score_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(results, ensure_ascii=False) + "\n")
    print(f"acc:{results}")


if __name__ == "__main__":
    main()
