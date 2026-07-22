"""Evaluate synthetic text-to-SQL predictions."""

import argparse
import copy
import json
import os
import re
import sqlite3
from collections import Counter
from typing import Any, Dict, Iterable, List, Sequence, Tuple


SQL_START_KEYWORDS = {
    "ALTER",
    "BEGIN",
    "COMMIT",
    "CREATE",
    "DELETE",
    "DROP",
    "EXPLAIN",
    "INSERT",
    "PRAGMA",
    "REPLACE",
    "ROLLBACK",
    "SELECT",
    "TRUNCATE",
    "UPDATE",
    "VACUUM",
    "WITH",
}
SQLITE_PROGRESS_INTERVAL = 10_000
SQLITE_MAX_PROGRESS_CALLBACKS = 10_000
SQL_CONTEXT_SEPARATOR = "\n\n### Context:\n"


def read_jsonl(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: str, rows: List[Dict]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _scan_words(text: str) -> Iterable[Tuple[str, int, int]]:
    """Yield unquoted SQL words as ``(word, offset, parenthesis_depth)``."""
    index = 0
    depth = 0
    length = len(text)
    while index < length:
        char = text[index]
        if char in "'\"`":
            quote = char
            index += 1
            while index < length:
                if text[index] == quote:
                    if index + 1 < length and text[index + 1] == quote:
                        index += 2
                        continue
                    index += 1
                    break
                index += 1
            continue
        if char == "[":
            index += 1
            while index < length:
                if text[index] == "]":
                    if index + 1 < length and text[index + 1] == "]":
                        index += 2
                        continue
                    index += 1
                    break
                index += 1
            continue
        if text.startswith("--", index):
            newline = text.find("\n", index + 2)
            index = length if newline < 0 else newline + 1
            continue
        if text.startswith("/*", index):
            end = text.find("*/", index + 2)
            index = length if end < 0 else end + 2
            continue
        if char == "(":
            depth += 1
            index += 1
            continue
        if char == ")":
            depth = max(0, depth - 1)
            index += 1
            continue
        if char.isalpha() or char == "_":
            start = index
            index += 1
            while index < length and (text[index].isalnum() or text[index] in "_$"):
                index += 1
            yield text[start:index].upper(), start, depth
            continue
        index += 1


def _strip_sql_wrapper(text: str) -> str:
    text = str(text or "").strip()
    fenced = re.search(
        r"```(?:[ \t]*sql)?[ \t]*\r?\n?(.*?)```",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if fenced:
        text = fenced.group(1).strip()
    text = re.sub(
        r"^\s*(?:(?:sql|query|answer)\s*:\s*)+",
        "",
        text,
        flags=re.IGNORECASE,
    )
    for word, offset, _ in _scan_words(text):
        if word in SQL_START_KEYWORDS:
            text = text[offset:]
            break
    return text.strip()


def _canonicalize_sql(text: str) -> str:
    """Fold case/whitespace outside quotes while preserving quoted values."""
    output: List[str] = []
    index = 0
    pending_space = False
    length = len(text)
    while index < length:
        char = text[index]
        if char.isspace():
            pending_space = bool(output)
            index += 1
            continue
        if pending_space:
            output.append(" ")
            pending_space = False
        if char in "'\"`":
            quote = char
            output.append(char)
            index += 1
            while index < length:
                char = text[index]
                output.append(char)
                index += 1
                if char == quote:
                    if index < length and text[index] == quote:
                        output.append(text[index])
                        index += 1
                        continue
                    break
            continue
        if char == "[":
            output.append(char)
            index += 1
            while index < length:
                char = text[index]
                output.append(char)
                index += 1
                if char == "]":
                    if index < length and text[index] == "]":
                        output.append(text[index])
                        index += 1
                        continue
                    break
            continue
        output.append(char.lower())
        index += 1

    normalized = "".join(output).strip()
    if normalized.endswith(";"):
        normalized = normalized[:-1].rstrip()
    return normalized


def normalize_sql(text: str) -> str:
    """Normalize the full SQL response without truncating any statements."""
    return _canonicalize_sql(_strip_sql_wrapper(text))


def sql_label(row: Dict) -> str:
    return normalize_sql(row.get("answer", "") or row.get("output", "") or row.get("sql", ""))


def _split_sql_statements(sql: str) -> List[str]:
    """Split complete SQLite statements, including trigger bodies, safely."""
    statements: List[str] = []
    buffer: List[str] = []
    for char in sql:
        buffer.append(char)
        if char == ";":
            candidate = "".join(buffer)
            if sqlite3.complete_statement(candidate):
                if candidate.strip():
                    statements.append(candidate.strip())
                buffer = []
    remainder = "".join(buffer).strip()
    if remainder:
        statements.append(remainder)
    return statements


def _has_top_level_order_by(statement: str) -> bool:
    words = [word for word, _, depth in _scan_words(statement) if depth == 0]
    return any(first == "ORDER" and second == "BY" for first, second in zip(words, words[1:]))


def _quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _database_snapshot(connection: sqlite3.Connection) -> Tuple[Tuple, Tuple]:
    schema_rows = connection.execute(
        "SELECT type, name, tbl_name, sql FROM sqlite_schema "
        "WHERE name NOT LIKE 'sqlite_%' ORDER BY type, name"
    ).fetchall()
    schema = tuple(
        (kind, name, table, _canonicalize_sql(definition or ""))
        for kind, name, table, definition in schema_rows
    )
    table_names = sorted(
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_schema "
            "WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
    )
    tables = []
    for name in table_names:
        rows = connection.execute(f"SELECT * FROM {_quote_identifier(name)}").fetchall()
        # Counter comparison retains duplicate rows without imposing an order on
        # the physical table scan.
        tables.append((name, Counter(tuple(row) for row in rows)))
    return schema, tuple(tables)


def _secure_connection() -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    denied_actions = {sqlite3.SQLITE_ATTACH, sqlite3.SQLITE_DETACH}

    def authorizer(action, arg1, arg2, _database, _trigger):
        function_name = str(arg2 or arg1 or "").lower()
        denied = action in denied_actions or (
            action == sqlite3.SQLITE_FUNCTION and function_name == "load_extension"
        )
        return sqlite3.SQLITE_DENY if denied else sqlite3.SQLITE_OK

    progress_calls = 0

    def progress_handler():
        nonlocal progress_calls
        progress_calls += 1
        return int(progress_calls > SQLITE_MAX_PROGRESS_CALLBACKS)

    connection.set_authorizer(authorizer)
    connection.set_progress_handler(progress_handler, SQLITE_PROGRESS_INTERVAL)
    return connection


def _new_context_connection(context: str) -> sqlite3.Connection:
    connection = _secure_connection()
    try:
        if context.strip():
            connection.executescript(context)
        connection.commit()
    except Exception:
        connection.close()
        raise
    return connection


def _execute_sql(
    connection: sqlite3.Connection, sql: str
) -> Tuple[List[Tuple[List[Tuple], bool, int]], Tuple]:
    statements = _split_sql_statements(sql)
    if not statements:
        raise sqlite3.OperationalError("empty SQL prediction")
    result_sets: List[Tuple[List[Tuple], bool, int]] = []
    for statement in statements:
        cursor = connection.execute(statement)
        if cursor.description is not None:
            result_sets.append(
                (
                    cursor.fetchall(),
                    _has_top_level_order_by(statement),
                    len(cursor.description),
                )
            )
    connection.commit()
    return result_sets, _database_snapshot(connection)


def _rows_match(gold_rows: Sequence[Tuple], pred_rows: Sequence[Tuple], ordered: bool) -> bool:
    if ordered:
        return list(gold_rows) == list(pred_rows)
    return Counter(tuple(row) for row in gold_rows) == Counter(tuple(row) for row in pred_rows)


def _execution_outputs_match(gold_output: Tuple, pred_output: Tuple) -> bool:
    gold_results, gold_snapshot = gold_output
    pred_results, pred_snapshot = pred_output
    if gold_snapshot != pred_snapshot or len(gold_results) != len(pred_results):
        return False
    for gold_result, pred_result in zip(gold_results, pred_results):
        gold_rows, gold_ordered, gold_width = gold_result
        pred_rows, _pred_ordered, pred_width = pred_result
        if gold_width != pred_width or not _rows_match(
            gold_rows, pred_rows, ordered=gold_ordered
        ):
            return False
    return True


def _context_for_row(row: Dict) -> str:
    context = str(row.get("sql_context", "") or row.get("context", "") or "")
    if context:
        return context
    input_text = str(row.get("input", "") or "")
    if SQL_CONTEXT_SEPARATOR in input_text:
        return input_text.split(SQL_CONTEXT_SEPARATOR, 1)[1]
    return ""


def evaluate_sql_metrics(rows: List[Dict]) -> Tuple[Dict[str, Any], List[Dict]]:
    checked = []
    exact_match_correct = 0
    execution_correct = 0
    execution_eligible = 0
    context_invalid = 0
    gold_invalid = 0
    prediction_invalid = 0

    for row in rows:
        pred_sql = _strip_sql_wrapper(row.get("response", ""))
        gold_source = row.get("answer", "") or row.get("output", "") or row.get("sql", "")
        gold_sql = _strip_sql_wrapper(gold_source)
        normalized_pred = _canonicalize_sql(pred_sql)
        normalized_gold = _canonicalize_sql(gold_sql)
        exact_flag = bool(normalized_pred) and normalized_pred == normalized_gold
        exact_match_correct += int(exact_flag)

        new_row = copy.deepcopy(row)
        new_row["pred"] = normalized_pred
        new_row["normalized_answer"] = normalized_gold
        new_row["flag"] = exact_flag
        new_row["execution_eligible"] = False
        new_row["execution_match"] = None

        context = _context_for_row(row)
        try:
            base = _new_context_connection(context)
        except Exception as error:
            context_invalid += 1
            new_row["execution_status"] = "context_invalid"
            new_row["execution_error"] = str(error)
            checked.append(new_row)
            continue

        gold_connection = _secure_connection()
        pred_connection = _secure_connection()
        try:
            base.backup(gold_connection)
            base.backup(pred_connection)
        finally:
            base.close()

        try:
            gold_output = _execute_sql(gold_connection, gold_sql)
        except Exception as error:
            gold_invalid += 1
            new_row["execution_status"] = "gold_invalid"
            new_row["execution_error"] = str(error)
            gold_connection.close()
            pred_connection.close()
            checked.append(new_row)
            continue

        execution_eligible += 1
        new_row["execution_eligible"] = True
        try:
            pred_output = _execute_sql(pred_connection, pred_sql)
        except Exception as error:
            prediction_invalid += 1
            new_row["execution_status"] = "prediction_invalid"
            new_row["execution_error"] = str(error)
        else:
            execution_match = _execution_outputs_match(gold_output, pred_output)
            execution_correct += int(execution_match)
            new_row["execution_match"] = execution_match
            new_row["execution_status"] = "match" if execution_match else "mismatch"
        finally:
            gold_connection.close()
            pred_connection.close()
        checked.append(new_row)

    total = len(rows)
    metrics = {
        "exact_match": exact_match_correct / total if total else 0.0,
        "exact_match_correct": exact_match_correct,
        "total": total,
        "execution_accuracy": (
            execution_correct / execution_eligible if execution_eligible else None
        ),
        "execution_correct": execution_correct,
        "execution_eligible": execution_eligible,
        "execution_coverage": execution_eligible / total if total else 0.0,
        "context_invalid": context_invalid,
        "gold_invalid": gold_invalid,
        "prediction_invalid": prediction_invalid,
    }
    return metrics, checked


def evaluate_sql(rows: List[Dict]) -> Tuple[float, List[Dict]]:
    """Compatibility wrapper returning only normalized exact match and rows."""
    metrics, checked = evaluate_sql_metrics(rows)
    return metrics["exact_match"], checked


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate synthetic text-to-SQL predictions."
    )
    parser.add_argument("--prediction_dir", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    benchmark = "synthetic_text_to_sql"
    path = os.path.join(args.prediction_dir, f"{benchmark}_responses.jsonl")
    rows = read_jsonl(path)
    metrics, checked = evaluate_sql_metrics(rows)
    primary = metrics["exact_match"]
    results: Dict[str, Any] = {
        benchmark: primary,
        f"{benchmark}_metrics": metrics,
        "average": primary,
    }
    write_jsonl(
        os.path.join(args.prediction_dir, f"{benchmark}_predict_checkanswer.jsonl"),
        checked,
    )
    print(f"{benchmark}: exact_match {primary:.6f}")

    results["prediction_dir"] = args.prediction_dir
    score_path = os.path.join(
        os.path.dirname(args.prediction_dir),
        "synthetic_text_to_sql_scores.jsonl",
    )
    with open(score_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(results, ensure_ascii=False) + "\n")
    print(f"acc:{results}")


if __name__ == "__main__":
    main()
