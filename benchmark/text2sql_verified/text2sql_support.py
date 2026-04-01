from __future__ import annotations

import copy
import functools
import importlib.util
import json
import multiprocessing as mp
import re
import sqlite3
import sys
import threading
import time
import types
import unicodedata
from pathlib import Path
from typing import Any

from app.bench.benchmark_support import public_question_payload
from app.codegen.verifier import load_callable_from_path


SQL_BLOCK_PATTERN = re.compile(r"```(?:sql)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
SQL_PREFIX_PATTERN = re.compile(r"^\s*(?:sql\s*:|answer\s*:)\s*", re.IGNORECASE)
QUESTION_MARKER = "Please write me a SQL statement that answers the following question:"
BIRD_EXECUTION_TIMEOUT_S = 30.0
BIRD_RESULT_PREVIEW_LIMIT = 10
SPIDER_EXECUTION_TIMEOUT_S = 30.0
SPIDER_RESULT_PREVIEW_LIMIT = 10
SPIDER_TOKEN_PATTERN = re.compile(
    r'"[^"\\]*(?:\\.[^"\\]*)*"|'
    r"'[^'\\]*(?:\\.[^'\\]*)*'|"
    r"!=|>=|<=|<>|"
    r"[A-Za-z0-9_\u4e00-\u9fff]+(?:\.[A-Za-z0-9_\u4e00-\u9fff]+)+|"
    r"[(),;=*<>+\-/]|"
    r"[A-Za-z0-9_\u4e00-\u9fff]+|"
    r"[0-9]+(?:\.[0-9]+)?|"
    r"[\u4e00-\u9fff]+"
)
_SPIDER_IMPORT_LOCK = threading.Lock()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    temp_path.replace(path)


def external_root(task_root: Path) -> Path:
    workspace_root = task_root.parents[2]
    return workspace_root / "".join(("ex", "ternal"))


def extract_sql_text(raw: object) -> str:
    if isinstance(raw, (list, tuple)):
        text = "\n".join(str(part) for part in raw if str(part).strip())
    else:
        text = str(raw or "")
    text = unicodedata.normalize("NFKC", text).strip()
    if not text:
        return ""

    block_match = SQL_BLOCK_PATTERN.search(text)
    if block_match:
        text = block_match.group(1).strip()

    text = SQL_PREFIX_PATTERN.sub("", text).strip()
    return text


def _normalize_sql_whitespace(text: str) -> str:
    pieces: list[str] = []
    token: list[str] = []
    quote: str | None = None
    i = 0
    while i < len(text):
        char = text[i]
        if quote is not None:
            token.append(char)
            if char == quote:
                if i + 1 < len(text) and text[i + 1] == quote:
                    token.append(text[i + 1])
                    i += 1
                else:
                    pieces.append("".join(token))
                    token = []
                    quote = None
            i += 1
            continue

        if char in {"'", '"', "`"}:
            if token:
                pieces.append("".join(token).lower())
                token = []
            quote = char
            token.append(char)
            i += 1
            continue

        if char.isspace():
            if token:
                pieces.append("".join(token).lower())
                token = []
            if not pieces or pieces[-1] != " ":
                pieces.append(" ")
            i += 1
            continue

        if char in {",", "(", ")", ";"}:
            if token:
                pieces.append("".join(token).lower())
                token = []
            if pieces and pieces[-1] == " ":
                pieces.pop()
            pieces.append(char)
            i += 1
            continue

        token.append(char)
        i += 1

    if token:
        fragment = "".join(token)
        pieces.append(fragment if quote is not None else fragment.lower())

    normalized = "".join(pieces)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    normalized = re.sub(r"\s*([(),;])\s*", r"\1", normalized)
    normalized = normalized.rstrip(";").strip()
    return normalized


def normalize_sql_text(value: object) -> str:
    sql = extract_sql_text(value)
    if not sql:
        return ""
    return _normalize_sql_whitespace(sql)


def sql_matches(actual: object, expected: object) -> tuple[bool, str]:
    actual_sql = normalize_sql_text(actual)
    expected_sql = normalize_sql_text(expected)
    return actual_sql == expected_sql and bool(expected_sql), actual_sql


def _task_root(task: dict[str, Any]) -> Path:
    task_dir = task.get("task_dir")
    if isinstance(task_dir, str) and task_dir.strip():
        return Path(task_dir).resolve()
    editable_path = task.get("editable_path")
    if isinstance(editable_path, str) and editable_path.strip():
        return Path(editable_path).resolve().parent
    return Path(__file__).resolve().parent


def _bird_db_path(task: dict[str, Any], item: dict[str, Any]) -> Path:
    metadata = dict(item.get("metadata") or {})
    raw_path = str(metadata.get("db_path") or "").strip()
    task_root = _task_root(task)
    if raw_path:
        db_path = Path(raw_path)
        return db_path if db_path.is_absolute() else (task_root / db_path).resolve()

    db_id = str(metadata.get("db_id") or "").strip()
    if not db_id:
        raise ValueError("BIRD execution evaluation requires metadata.db_id or metadata.db_path.")
    return (task_root / "data" / "dev_databases" / db_id / f"{db_id}.sqlite").resolve()


def _chase_db_path(task: dict[str, Any], item: dict[str, Any]) -> Path:
    metadata = dict(item.get("metadata") or {})
    raw_path = str(metadata.get("db_path") or "").strip()
    task_root = _task_root(task)
    if raw_path:
        db_path = Path(raw_path)
        return db_path if db_path.is_absolute() else (task_root / db_path).resolve()

    db_id = str(metadata.get("db_id") or "").strip()
    if not db_id:
        raise ValueError("CHASE execution evaluation requires metadata.db_id or metadata.db_path.")
    return (task_root / "data" / "database" / f"{db_id}.sqlite").resolve()


def _bird_execute_sql_worker(
    db_path: str,
    predicted_sql: str,
    gold_sql: str,
    queue: Any,
) -> None:
    payload: dict[str, Any]
    try:
        connection = sqlite3.connect(db_path)
        try:
            cursor = connection.cursor()
            try:
                cursor.execute(predicted_sql)
                predicted_rows = cursor.fetchall()
            except Exception as exc:  # noqa: BLE001
                payload = {
                    "state": "predicted_error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            else:
                try:
                    cursor.execute(gold_sql)
                    gold_rows = cursor.fetchall()
                except Exception as exc:  # noqa: BLE001
                    payload = {
                        "state": "gold_error",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                else:
                    payload = {
                        "state": "ok",
                        "passed": int(set(predicted_rows) == set(gold_rows)),
                        "predicted_preview": predicted_rows[:BIRD_RESULT_PREVIEW_LIMIT],
                        "gold_preview": gold_rows[:BIRD_RESULT_PREVIEW_LIMIT],
                        "predicted_count": len(predicted_rows),
                        "gold_count": len(gold_rows),
                    }
        finally:
            connection.close()
    except Exception as exc:  # noqa: BLE001
        payload = {
            "state": "runtime_error",
            "error": f"{type(exc).__name__}: {exc}",
        }
    queue.put(payload)


def execute_bird_sqls(
    *,
    predicted_sql: str,
    gold_sql: str,
    db_path: Path,
    timeout_s: float = BIRD_EXECUTION_TIMEOUT_S,
) -> dict[str, Any]:
    context = mp.get_context("spawn")
    queue = context.Queue()
    process = context.Process(
        target=_bird_execute_sql_worker,
        args=(str(db_path), predicted_sql, gold_sql, queue),
    )
    process.start()
    process.join(timeout_s)
    if process.is_alive():
        process.kill()
        process.join()
        return {
            "state": "timeout",
            "error": f"SQL execution exceeded {timeout_s:.1f}s timeout.",
        }
    if process.exitcode not in {0, None} and queue.empty():
        return {
            "state": "runtime_error",
            "error": f"SQL execution worker exited with code {process.exitcode}.",
        }
    if queue.empty():
        return {
            "state": "runtime_error",
            "error": "SQL execution worker returned no payload.",
        }
    return dict(queue.get())


def _spider_result_map(rows: list[tuple[Any, ...]], select_units: list[Any]) -> dict[Any, list[Any]]:
    result_map: dict[Any, list[Any]] = {}
    for index, unit in enumerate(select_units):
        val_unit = unit[1]
        if val_unit[2]:
            key = (val_unit[0], tuple(val_unit[1]), tuple(val_unit[2]))
        else:
            key = tuple(val_unit[1])
        result_map[key] = [row[index] for row in rows]
    return result_map


def _spider_execute_sql_worker(
    spider_root: str,
    db_path: str,
    predicted_sql: str,
    gold_sql: str,
    queue: Any,
) -> None:
    payload: dict[str, Any]
    try:
        process_module, _ = _load_spider_evaluation_modules(spider_root)
        schema = process_module.Schema(process_module.get_schema(db_path))
        try:
            predicted = process_module.get_sql(schema, predicted_sql)
        except Exception as exc:  # noqa: BLE001
            payload = {
                "state": "predicted_parse_error",
                "error": f"{type(exc).__name__}: {exc}",
            }
        else:
            try:
                gold = process_module.get_sql(schema, gold_sql)
            except Exception as exc:  # noqa: BLE001
                payload = {
                    "state": "gold_parse_error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            else:
                connection = sqlite3.connect(db_path)
                try:
                    cursor = connection.cursor()
                    try:
                        cursor.execute(predicted_sql)
                        predicted_rows = cursor.fetchall()
                    except Exception as exc:  # noqa: BLE001
                        payload = {
                            "state": "predicted_error",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    else:
                        try:
                            cursor.execute(gold_sql)
                            gold_rows = cursor.fetchall()
                        except Exception as exc:  # noqa: BLE001
                            payload = {
                                "state": "gold_error",
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        else:
                            payload = {
                                "state": "ok",
                                "passed": int(
                                    _spider_result_map(predicted_rows, predicted["select"][1])
                                    == _spider_result_map(gold_rows, gold["select"][1])
                                ),
                                "predicted_preview": predicted_rows[:SPIDER_RESULT_PREVIEW_LIMIT],
                                "gold_preview": gold_rows[:SPIDER_RESULT_PREVIEW_LIMIT],
                                "predicted_count": len(predicted_rows),
                                "gold_count": len(gold_rows),
                            }
                finally:
                    connection.close()
    except Exception as exc:  # noqa: BLE001
        payload = {
            "state": "runtime_error",
            "error": f"{type(exc).__name__}: {exc}",
        }
    queue.put(payload)


def execute_spider_sqls(
    *,
    spider_root: Path,
    predicted_sql: str,
    gold_sql: str,
    db_path: Path,
    timeout_s: float = SPIDER_EXECUTION_TIMEOUT_S,
) -> dict[str, Any]:
    context = mp.get_context("spawn")
    queue = context.Queue()
    process = context.Process(
        target=_spider_execute_sql_worker,
        args=(str(spider_root), str(db_path), predicted_sql, gold_sql, queue),
    )
    process.start()
    process.join(timeout_s)
    if process.is_alive():
        process.kill()
        process.join()
        return {
            "state": "timeout",
            "error": f"SQL execution exceeded {timeout_s:.1f}s timeout.",
        }
    if process.exitcode not in {0, None} and queue.empty():
        return {
            "state": "runtime_error",
            "error": f"SQL execution worker exited with code {process.exitcode}.",
        }
    if queue.empty():
        return {
            "state": "runtime_error",
            "error": "SQL execution worker returned no payload.",
        }
    return dict(queue.get())


def schema_text_from_table_entry(entry: dict[str, Any]) -> str:
    table_names = list(entry.get("table_names_original") or entry.get("table_names") or [])
    column_names = list(entry.get("column_names_original") or entry.get("column_names") or [])
    column_types = list(entry.get("column_types") or [])
    primary_keys = set(entry.get("primary_keys") or [])
    foreign_keys = list(entry.get("foreign_keys") or [])

    columns_by_table: dict[int, list[str]] = {index: [] for index, _ in enumerate(table_names)}
    for column_index, column in enumerate(column_names):
        if not isinstance(column, list) or len(column) != 2:
            continue
        table_index, column_name = column
        if table_index is None or int(table_index) < 0:
            continue
        type_name = "TEXT"
        if column_index < len(column_types):
            type_name = str(column_types[column_index] or "TEXT").upper()
        parts = [str(column_name), "[", type_name, "]"]
        if column_index in primary_keys:
            parts.append("primary_key")
        columns_by_table[int(table_index)].append(" ".join(parts))

    relation_lines: list[str] = []
    for source_index, target_index in foreign_keys:
        try:
            source_table_index, source_name = column_names[source_index]
            target_table_index, target_name = column_names[target_index]
            source_table = table_names[source_table_index]
            target_table = table_names[target_table_index]
        except Exception:
            continue
        relation_lines.append(
            f"{source_table}.{source_name} = {target_table}.{target_name}"
        )

    lines: list[str] = []
    for table_index, table_name in enumerate(table_names):
        lines.append(f"{table_name} :")
        for column_line in columns_by_table.get(table_index, []):
            lines.append(column_line)
        lines.append("")
    if relation_lines:
        lines.append("Relations:")
        lines.extend(relation_lines)
    return "\n".join(lines).strip()


def evaluate_sql_candidate(*, task, candidate_path):
    item = task.get("question_item")
    if not isinstance(item, dict):
        raise ValueError("Dataset question task must provide question_item.")

    started = time.perf_counter()
    solver = load_callable_from_path(candidate_path, str(task["entry_symbol"]))
    raw_actual = solver(public_question_payload(item))
    passed, actual = sql_matches(raw_actual, item["expected_answer"])
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    row = {
        "name": item.get("name") or item["item_id"],
        "expected": item["expected_answer"],
        "actual": actual,
        "actual_raw": str(raw_actual or ""),
        "passed": passed,
    }
    return {
        "status": "pass" if passed else "fail",
        "verifier_status": "pass" if passed else "fail",
        "correctness": 1.0 if passed else 0.0,
        "passed_tests": 1 if passed else 0,
        "total_tests": 1,
        "benchmark_ms": round(elapsed_ms, 3),
        "benchmark_samples_ms": [round(elapsed_ms, 3)],
        "objective": 1.0 if passed else 0.0,
        "objective_score": 1.0 if passed else 0.0,
        "objective_signal": 1.0 if passed else 0.0,
        "error": None,
        "test_results": [row],
    }


def evaluate_bird_execution_candidate(*, task, candidate_path):
    item = task.get("question_item")
    if not isinstance(item, dict):
        raise ValueError("Dataset question task must provide question_item.")

    db_path = _bird_db_path(task, item)
    if not db_path.exists():
        raise FileNotFoundError(f"BIRD sqlite database not found: {db_path}")

    started = time.perf_counter()
    solver = load_callable_from_path(candidate_path, str(task["entry_symbol"]))
    raw_actual = solver(public_question_payload(item))
    actual_sql = extract_sql_text(raw_actual)
    gold_sql = extract_sql_text(item.get("raw_expected_answer") or item["expected_answer"])
    execution = execute_bird_sqls(predicted_sql=actual_sql, gold_sql=gold_sql, db_path=db_path)
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    state = str(execution.get("state") or "runtime_error")
    passed = state == "ok" and bool(execution.get("passed"))
    if state == "gold_error":
        raise RuntimeError(
            f"BIRD gold SQL failed to execute for {item.get('item_id') or '<unknown>'} on {db_path}: "
            f"{execution.get('error') or 'unknown error'}"
        )

    row = {
        "name": item.get("name") or item["item_id"],
        "expected": gold_sql,
        "actual": actual_sql,
        "actual_raw": str(raw_actual or ""),
        "passed": passed,
        "comparison_mode": "bird_execution",
        "db_path": str(db_path),
        "db_id": str(dict(item.get("metadata") or {}).get("db_id") or ""),
    }
    if execution.get("error"):
        row["execution_error"] = str(execution["error"])
    if execution.get("predicted_preview") is not None:
        row["predicted_result_preview"] = execution.get("predicted_preview")
    if execution.get("gold_preview") is not None:
        row["expected_result_preview"] = execution.get("gold_preview")
    if execution.get("predicted_count") is not None:
        row["predicted_result_count"] = execution.get("predicted_count")
    if execution.get("gold_count") is not None:
        row["expected_result_count"] = execution.get("gold_count")

    return {
        "status": "pass" if passed else "fail",
        "verifier_status": "pass" if passed else "fail",
        "correctness": 1.0 if passed else 0.0,
        "passed_tests": 1 if passed else 0,
        "total_tests": 1,
        "benchmark_ms": round(elapsed_ms, 3),
        "benchmark_samples_ms": [round(elapsed_ms, 3)],
        "objective": 1.0 if passed else 0.0,
        "objective_score": 1.0 if passed else 0.0,
        "objective_signal": 1.0 if passed else 0.0,
        "error": None if passed else row.get("execution_error"),
        "test_results": [row],
    }


def _spider_word_tokenize(text: str) -> list[str]:
    return SPIDER_TOKEN_PATTERN.findall(str(text))


@functools.lru_cache(maxsize=4)
def _load_spider_evaluation_modules(spider_root: str) -> tuple[Any, Any]:
    with _SPIDER_IMPORT_LOCK:
        root = Path(spider_root).resolve()
        nltk_module = sys.modules.get("nltk")
        if nltk_module is None:
            nltk_module = types.ModuleType("nltk")
            sys.modules["nltk"] = nltk_module
        if not hasattr(nltk_module, "word_tokenize"):
            nltk_module.word_tokenize = _spider_word_tokenize

        process_spec = importlib.util.spec_from_file_location("process_sql", root / "process_sql.py")
        if process_spec is None or process_spec.loader is None:
            raise ImportError(f"Unable to load Spider process_sql.py from {root}.")
        process_module = importlib.util.module_from_spec(process_spec)
        sys.modules["process_sql"] = process_module
        process_spec.loader.exec_module(process_module)

        evaluation_spec = importlib.util.spec_from_file_location("spider_evaluation", root / "evaluation.py")
        if evaluation_spec is None or evaluation_spec.loader is None:
            raise ImportError(f"Unable to load Spider evaluation.py from {root}.")
        evaluation_module = importlib.util.module_from_spec(evaluation_spec)
        evaluation_spec.loader.exec_module(evaluation_module)
        return process_module, evaluation_module


def _schema_dict_from_table_entry(entry: dict[str, Any]) -> dict[str, list[str]]:
    table_names = [str(name).lower() for name in entry.get("table_names_original") or []]
    schema: dict[str, list[str]] = {table_name: [] for table_name in table_names}
    for column in entry.get("column_names_original") or []:
        if not isinstance(column, list) or len(column) != 2:
            continue
        table_index, column_name = column
        if table_index is None or int(table_index) < 0:
            continue
        table_name = table_names[int(table_index)]
        schema.setdefault(table_name, []).append(str(column_name).lower())
    return schema


def _chase_source_dir(task: dict[str, Any]) -> Path:
    task_root = _task_root(task)
    local_source = task_root / "data" / "source"
    if local_source.exists():
        return local_source
    return external_root(task_root) / "chase-dataset" / "data"


@functools.lru_cache(maxsize=8)
def _load_chase_tables(source_dir: str) -> dict[str, dict[str, Any]]:
    tables = json.loads((Path(source_dir) / "tables.json").read_text())
    return {
        str(entry.get("db_id") or ""): dict(entry)
        for entry in tables
        if isinstance(entry, dict) and str(entry.get("db_id") or "").strip()
    }


def _empty_spider_sql() -> dict[str, Any]:
    return {
        "except": None,
        "from": {"conds": [], "table_units": []},
        "groupBy": [],
        "having": [],
        "intersect": None,
        "limit": None,
        "orderBy": [],
        "select": [False, []],
        "union": None,
        "where": [],
    }


def _normalize_spider_sql(
    *,
    sql_text: str,
    schema: Any,
    table_entry: dict[str, Any],
    process_module: Any,
    evaluation_module: Any,
    allow_empty_on_error: bool,
) -> tuple[dict[str, Any], str | None]:
    try:
        parsed = process_module.get_sql(schema, sql_text)
    except Exception as exc:  # noqa: BLE001
        if not allow_empty_on_error:
            raise
        return _empty_spider_sql(), f"{type(exc).__name__}: {exc}"

    valid_col_units = evaluation_module.build_valid_col_units(parsed["from"]["table_units"], schema)
    normalized = evaluation_module.rebuild_sql_val(copy.deepcopy(parsed))
    normalized = evaluation_module.rebuild_sql_col(
        valid_col_units,
        normalized,
        evaluation_module.build_foreign_key_map(table_entry),
    )
    return normalized, None


def evaluate_chase_spider_candidate(*, task, candidate_path):
    item = task.get("question_item")
    if not isinstance(item, dict):
        raise ValueError("Dataset question task must provide question_item.")

    metadata = dict(item.get("metadata") or {})
    db_id = str(metadata.get("db_id") or "").strip()
    if not db_id:
        raise ValueError("CHASE evaluation requires metadata.db_id.")

    task_root = _task_root(task)
    spider_root = external_root(task_root) / "spider"
    process_module, evaluation_module = _load_spider_evaluation_modules(str(spider_root))

    source_dir = _chase_source_dir(task)
    tables_by_db = _load_chase_tables(str(source_dir))
    table_entry = tables_by_db.get(db_id)
    if table_entry is None:
        raise KeyError(f"CHASE tables.json is missing db_id={db_id!r}.")

    schema = process_module.Schema(_schema_dict_from_table_entry(table_entry))

    started = time.perf_counter()
    solver = load_callable_from_path(candidate_path, str(task["entry_symbol"]))
    raw_actual = solver(public_question_payload(item))
    actual_sql = extract_sql_text(raw_actual)
    gold_sql = extract_sql_text(item.get("raw_expected_answer") or item["expected_answer"])

    gold_parsed, _ = _normalize_spider_sql(
        sql_text=gold_sql,
        schema=schema,
        table_entry=table_entry,
        process_module=process_module,
        evaluation_module=evaluation_module,
        allow_empty_on_error=False,
    )
    predicted_parsed, parse_error = _normalize_spider_sql(
        sql_text=actual_sql,
        schema=schema,
        table_entry=table_entry,
        process_module=process_module,
        evaluation_module=evaluation_module,
        allow_empty_on_error=True,
    )
    evaluator = evaluation_module.Evaluator()
    passed = bool(evaluator.eval_exact_match(predicted_parsed, gold_parsed))
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    row = {
        "name": item.get("name") or item["item_id"],
        "expected": gold_sql,
        "actual": actual_sql,
        "actual_raw": str(raw_actual or ""),
        "passed": passed,
        "comparison_mode": "spider_exact_match",
        "db_id": db_id,
        "partial_scores": evaluator.partial_scores,
    }
    if parse_error:
        row["parse_error"] = parse_error

    return {
        "status": "pass" if passed else "fail",
        "verifier_status": "pass" if passed else "fail",
        "correctness": 1.0 if passed else 0.0,
        "passed_tests": 1 if passed else 0,
        "total_tests": 1,
        "benchmark_ms": round(elapsed_ms, 3),
        "benchmark_samples_ms": [round(elapsed_ms, 3)],
        "objective": 1.0 if passed else 0.0,
        "objective_score": 1.0 if passed else 0.0,
        "objective_signal": 1.0 if passed else 0.0,
        "error": parse_error,
        "test_results": [row],
    }


def evaluate_chase_execution_candidate(*, task, candidate_path):
    item = task.get("question_item")
    if not isinstance(item, dict):
        raise ValueError("Dataset question task must provide question_item.")

    metadata = dict(item.get("metadata") or {})
    db_id = str(metadata.get("db_id") or "").strip()
    if not db_id:
        raise ValueError("CHASE evaluation requires metadata.db_id.")

    db_path = _chase_db_path(task, item)
    if not db_path.exists():
        raise FileNotFoundError(f"CHASE sqlite database not found: {db_path}")

    task_root = _task_root(task)
    spider_root = external_root(task_root) / "spider"

    started = time.perf_counter()
    solver = load_callable_from_path(candidate_path, str(task["entry_symbol"]))
    raw_actual = solver(public_question_payload(item))
    actual_sql = extract_sql_text(raw_actual)
    gold_sql = extract_sql_text(item.get("raw_expected_answer") or item["expected_answer"])
    execution = execute_spider_sqls(
        spider_root=spider_root,
        predicted_sql=actual_sql,
        gold_sql=gold_sql,
        db_path=db_path,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    state = str(execution.get("state") or "runtime_error")
    passed = state == "ok" and bool(execution.get("passed"))
    if state in {"gold_error", "gold_parse_error"}:
        raise RuntimeError(
            f"CHASE gold SQL failed to execute for {item.get('item_id') or '<unknown>'} on {db_path}: "
            f"{execution.get('error') or 'unknown error'}"
        )

    row = {
        "name": item.get("name") or item["item_id"],
        "expected": gold_sql,
        "actual": actual_sql,
        "actual_raw": str(raw_actual or ""),
        "passed": passed,
        "comparison_mode": "spider_execution",
        "db_id": db_id,
        "db_path": str(db_path),
    }
    if execution.get("error"):
        row["execution_error"] = str(execution["error"])
    if execution.get("predicted_preview") is not None:
        row["predicted_result_preview"] = execution.get("predicted_preview")
    if execution.get("gold_preview") is not None:
        row["expected_result_preview"] = execution.get("gold_preview")
    if execution.get("predicted_count") is not None:
        row["predicted_result_count"] = execution.get("predicted_count")
    if execution.get("gold_count") is not None:
        row["expected_result_count"] = execution.get("gold_count")

    return {
        "status": "pass" if passed else "fail",
        "verifier_status": "pass" if passed else "fail",
        "correctness": 1.0 if passed else 0.0,
        "passed_tests": 1 if passed else 0,
        "total_tests": 1,
        "benchmark_ms": round(elapsed_ms, 3),
        "benchmark_samples_ms": [round(elapsed_ms, 3)],
        "objective": 1.0 if passed else 0.0,
        "objective_score": 1.0 if passed else 0.0,
        "objective_signal": 1.0 if passed else 0.0,
        "error": None if passed else row.get("execution_error"),
        "test_results": [row],
    }


def parse_bird_prompt(prompt: str) -> tuple[str, str]:
    text = unicodedata.normalize("NFKC", str(prompt or "")).strip()
    if not text:
        return "", ""
    body = text
    if body.startswith("[INST]"):
        body = body[len("[INST]") :].strip()
    if body.endswith("[/INST]"):
        body = body[: -len("[/INST]")].strip()
    if QUESTION_MARKER not in body:
        return body, ""
    schema, question = body.rsplit(QUESTION_MARKER, 1)
    schema = schema.replace("Here is a database schema:", "", 1).strip()
    return schema, question.strip()
