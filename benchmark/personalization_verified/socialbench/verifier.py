from __future__ import annotations

import re
import time

from app.bench.benchmark_support import public_question_payload
from app.codegen.verifier import load_callable_from_path


ANSWER_PATTERN = re.compile(r"(\b|\W+|^|[\u4e00-\u9fa5]+|(?<=[A-D]))([A-H])(\b|(?=[A-D])|$|\W+|[\u4e00-\u9fa5]+)")


def _canonical_category(category: str) -> str:
    # The released data uses Individual-MEM-Short/Long while upstream dataset.py
    # keys the open-ended prompt/scoring path off the shared Individual-MEM base.
    if category.startswith("Individual-MEM"):
        return "Individual-MEM"
    return category


def _format_predict(predict: str | None) -> list[str] | None:
    if predict is None:
        return None
    answers: list[str] = []
    matches = ANSWER_PATTERN.findall(predict)
    for match in matches:
        candidate = match[1]
        if candidate not in answers:
            answers.append(candidate)
    return answers


def _score_prediction(actual: str, labels: list[str], category: str | None = None) -> float | None:
    if _canonical_category(category or "") == "Individual-MEM":
        lowered = actual.lower()
        if len(lowered) == 0:
            return None
        score = 0
        for keyword in labels:
            score += 1 if str(keyword).lower() in lowered else 0
        return score / len(labels)

    answers = _format_predict(actual)
    if not answers:
        return None
    if len(labels) == 1:
        return 1.0 if answers[0] == labels[0] else 0.0
    for answer in answers:
        if answer not in labels:
            return 0.0
    return len(set(answers)) / len(set(labels))


def evaluate_candidate(*, task, candidate_path, source_code, baseline_metrics, memory_applied):
    del source_code, baseline_metrics, memory_applied
    item = task.get("question_item")
    if not isinstance(item, dict):
        raise ValueError("Dataset question task must provide question_item.")

    category = _canonical_category(str(item.get("metadata", {}).get("category") or "").strip())
    labels = [str(value).strip() for value in list(item.get("expected_answer") or []) if str(value).strip()]

    started = time.perf_counter()
    solver = load_callable_from_path(candidate_path, str(task["entry_symbol"]))
    raw_actual = solver(public_question_payload(item))
    actual_text = str(raw_actual or "").strip()
    raw_score = _score_prediction(actual_text, labels, category)
    score = 0.0 if raw_score is None else float(raw_score)
    exact = score >= 1.0 - 1e-9
    well_formed = raw_score is not None
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    row = {
        "name": item.get("name") or item["item_id"],
        "expected": labels,
        "actual": actual_text,
        "actual_raw": actual_text,
        "passed": exact,
        "score": score,
    }
    return {
        "status": "pass" if well_formed else "fail",
        "verifier_status": "pass" if well_formed else "fail",
        "correctness": score,
        "passed_tests": 1 if well_formed else 0,
        "total_tests": 1,
        "benchmark_ms": round(elapsed_ms, 3),
        "benchmark_samples_ms": [round(elapsed_ms, 3)],
        "objective": score,
        "objective_score": score,
        "objective_signal": score,
        "error": None,
        "test_results": [row],
    }
