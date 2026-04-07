from __future__ import annotations

from difflib import SequenceMatcher
from pathlib import Path
import time

from app.bench.benchmark_support import normalize_answer_text, public_question_payload
from app.bench.personalization_support import mean_score, run_eval_model_json
from app.codegen.verifier import load_callable_from_path


def _lexical_similarity(expected: str, actual: str) -> float:
    normalized_expected = normalize_answer_text(expected)
    normalized_actual = normalize_answer_text(actual)
    if not normalized_expected or not normalized_actual:
        return 0.0
    return SequenceMatcher(None, normalized_actual, normalized_expected).ratio()


def _judge_payload(task: dict, item: dict, actual_text: str) -> tuple[dict, dict]:
    metadata = dict(item.get("metadata") or {})
    speaker = str(metadata.get("speaker") or "").strip() or "The target speaker"
    return run_eval_model_json(
        task=task,
        purpose=f"coser_judge:{item.get('item_id')}",
        system_prompt=(
            "You are grading a hidden CoSER literary-dialogue proxy. "
            "Return strict JSON only with 0-5 scores."
        ),
        user_prompt=(
            f"Target speaker: {speaker}\n"
            f"Book: {metadata.get('book') or ''}\n"
            f"Topic: {metadata.get('topic') or ''}\n\n"
            f"Scene and dialogue context:\n{item.get('context')}\n\n"
            f"Reference continuation:\n{item.get('expected_answer')}\n\n"
            f"Candidate continuation:\n{actual_text}\n\n"
            "Score these aspects from 0 to 5:\n"
            "- character_fidelity: does the continuation sound like the target character?\n"
            "- scene_grounding: does it stay grounded in the current literary scene and plot context?\n"
            "- dialogue_quality: is it a natural, coherent next turn?\n\n"
            'Return JSON like {"scores": {"character_fidelity": 4, "scene_grounding": 5, "dialogue_quality": 4}, "summary": "..."}'
        ),
    )


def evaluate_candidate(*, task, candidate_path, source_code, baseline_metrics, memory_applied):
    del source_code, baseline_metrics, memory_applied
    item = task.get("question_item")
    if not isinstance(item, dict):
        raise ValueError("Dataset question task must provide question_item.")

    started = time.perf_counter()
    solver = load_callable_from_path(Path(candidate_path), str(task["entry_symbol"]))
    raw_actual = solver(public_question_payload(item))
    actual_text = str(raw_actual or "").strip()
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
                    "expected": "non-empty literary continuation",
                    "actual": actual_text,
                    "actual_raw": str(raw_actual or ""),
                    "passed": False,
                }
            ],
        }

    expected = str(item.get("expected_answer") or "").strip()
    normalized_expected = normalize_answer_text(expected)
    normalized_actual = normalize_answer_text(actual_text)
    exact = bool(normalized_actual) and normalized_actual == normalized_expected
    lexical_similarity = _lexical_similarity(expected, actual_text)

    judge_scores: dict[str, float] = {}
    judge_average = None
    trace = None
    if task.get("eval_model"):
        payload, trace = _judge_payload(task, item, actual_text)
        raw_scores = payload.get("scores") if isinstance(payload, dict) else {}
        if isinstance(raw_scores, dict):
            for key, value in raw_scores.items():
                try:
                    judge_scores[str(key)] = max(0.0, min(5.0, float(value)))
                except (TypeError, ValueError):
                    continue
        judge_average = mean_score(list(judge_scores.values()), minimum=0.0, maximum=5.0)

    objective = 1.0 if exact else lexical_similarity
    if judge_average is not None and not exact:
        objective = round((0.25 * lexical_similarity) + (0.75 * (judge_average / 5.0)), 6)

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
                "expected": expected,
                "actual": actual_text,
                "actual_raw": actual_text,
                "passed": objective > 0.0,
                "exact_match": exact,
                "lexical_similarity": round(lexical_similarity, 6),
                "judge_average": round(judge_average, 6) if judge_average is not None else None,
                "judge_scores": judge_scores or None,
            }
        ],
        "judge_trace": trace,
    }
