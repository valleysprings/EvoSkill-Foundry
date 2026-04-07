from __future__ import annotations

from pathlib import Path
import time

from app.bench.benchmark_support import normalize_answer_text, public_question_payload
from app.bench.personalization_support import mean_score, run_eval_model_json
from app.codegen.verifier import load_callable_from_path


def _first_utterance(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return text.splitlines()[0].strip()


def evaluate_candidate(*, task, candidate_path, source_code, baseline_metrics, memory_applied):
    del source_code, baseline_metrics, memory_applied
    item = task.get("question_item")
    if not isinstance(item, dict):
        raise ValueError("Dataset question task must provide question_item.")

    started = time.perf_counter()
    solver = load_callable_from_path(Path(candidate_path), str(task["entry_symbol"]))
    raw_actual = solver(public_question_payload(item))
    actual_text = _first_utterance(raw_actual)
    if not actual_text:
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return {
            "status": "fail",
            "verifier_status": "fail",
            "correctness": 0.0,
            "passed_tests": 0,
            "total_tests": 1,
            "benchmark_ms": round(elapsed_ms, 3),
            "benchmark_samples_ms": [round(elapsed_ms, 3)],
            "objective": 0.0,
            "objective_score": 0.0,
            "objective_signal": 0.0,
            "error": None,
            "test_results": [
                {
                    "name": item.get("name") or item["item_id"],
                    "expected": "non-empty judged response",
                    "actual": actual_text,
                    "actual_raw": str(raw_actual or ""),
                    "passed": False,
                }
            ],
        }

    metrics = [str(value).strip() for value in list(item.get("metadata", {}).get("charactereval_metrics") or []) if str(value).strip()]
    payload, trace = run_eval_model_json(
        task=task,
        purpose=f"charactereval_judge:{item.get('item_id')}",
        system_prompt=(
            "You are grading a CharacterEval role-playing response. "
            "Return strict JSON only. Score each requested metric from 0 to 4, where 0 is very poor and 4 is excellent."
        ),
        user_prompt=(
            f"Context:\n{item.get('context')}\n\n"
            f"Candidate response:\n{actual_text}\n\n"
            f"Requested metrics: {', '.join(metrics) if metrics else 'overall quality'}\n\n"
            'Return JSON like {"scores": {"Accuracy": 4, "Behavior": 3}, "summary": "..."}'
        ),
    )
    raw_scores = payload.get("scores") if isinstance(payload, dict) else {}
    metric_scores: dict[str, float] = {}
    requested_metrics = {normalize_answer_text(metric): metric for metric in metrics}
    if isinstance(raw_scores, dict):
        for key, value in raw_scores.items():
            normalized_key = normalize_answer_text(key)
            if requested_metrics and normalized_key not in requested_metrics:
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            metric_name = requested_metrics.get(normalized_key, str(key))
            metric_scores[str(metric_name)] = max(0.0, min(4.0, numeric))
    overall = mean_score(metric_scores.values(), minimum=0.0, maximum=4.0)
    objective = 0.0 if overall is None else round(overall / 4.0, 6)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return {
        "status": "pass",
        "verifier_status": "pass",
        "correctness": objective,
        "passed_tests": 1,
        "total_tests": 1,
        "benchmark_ms": round(elapsed_ms, 3),
        "benchmark_samples_ms": [round(elapsed_ms, 3)],
        "objective": objective,
        "objective_score": objective,
        "objective_signal": objective,
        "error": None,
        "test_results": [
            {
                "name": item.get("name") or item["item_id"],
                "expected": "eval-model score",
                "actual": actual_text,
                "actual_raw": actual_text,
                "passed": objective > 0.0,
                "metric_scores": metric_scores,
                "judge_summary": payload.get("summary") if isinstance(payload, dict) else None,
            }
        ],
        "judge_trace": trace,
    }
