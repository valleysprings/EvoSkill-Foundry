from __future__ import annotations

import time

from app.bench.benchmark_support import normalize_answer_text, public_question_payload
from app.codegen.verifier import load_callable_from_path


def _stringify_answer(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return "\n".join(str(part) for part in value)
    return str(value)


def _match_answer(item: dict, raw_actual: object) -> tuple[bool, str]:
    prediction = _stringify_answer(raw_actual).strip()
    if not prediction:
        return False, ""
    normalized_prediction = normalize_answer_text(prediction)
    normalized_expected = normalize_answer_text(item["expected_answer"])
    return normalized_prediction == normalized_expected, normalized_prediction


def evaluate_candidate(*, task, candidate_path, source_code, baseline_metrics, memory_applied):
    item = task.get("question_item")
    if not isinstance(item, dict):
        raise ValueError("Dataset question task must provide question_item.")

    started = time.perf_counter()
    solver = load_callable_from_path(candidate_path, str(task["entry_symbol"]))
    raw_actual = solver(public_question_payload(item))
    passed, actual = _match_answer(item, raw_actual)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    row = {
        "name": item.get("name") or item["item_id"],
        "expected": item["expected_answer"],
        "actual": actual,
        "actual_raw": _stringify_answer(raw_actual),
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
