from __future__ import annotations

import re
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
TRACK_ROOT = ROOT.parent
REPO_ROOT = ROOT.parents[2]
for candidate in (TRACK_ROOT, REPO_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from sync_external import ensure_repo_checkout
from app.bench.benchmark_support import public_question_payload
from app.bench.personalization_support import run_eval_model_json
from app.codegen.verifier import load_callable_from_path


def _prompt_template(name: str) -> str:
    source_root = ensure_repo_checkout("timechara")
    path = source_root / "data" / name
    return path.read_text()


def _extract_score(content: str) -> float | None:
    last_sentence = content.strip().splitlines()[-1] if content.strip() else ""
    matches = re.findall(r"\b[0-7]\b", last_sentence)
    if not matches:
        return None
    return float(matches[-1])


def _judge_score(task: dict[str, Any], prompt_template: str, *, agent_name: str, question: str, answer: str, fact: str, purpose: str) -> tuple[float | None, dict[str, Any]]:
    payload, trace = run_eval_model_json(
        task=task,
        purpose=purpose,
        system_prompt=(
            "You are a TimeChara evaluator. Follow the official scoring rubric, but return strict JSON only as "
            '{"score": 0 or 1, "reasoning": "..."} .'
        ),
        user_prompt=(
            prompt_template.format(agent_name=agent_name, question_0=question, answer_0=answer, agent_fact_0=fact)
            + "\n\nReturn strict JSON only with keys score and reasoning."
        ),
    )
    reasoning = ""
    if isinstance(payload, dict):
        reasoning = str(payload.get("text") or payload.get("reasoning") or payload.get("content") or "").strip()
        raw_score = payload.get("score")
        if raw_score is not None:
            try:
                return float(raw_score), trace
            except (TypeError, ValueError):
                pass
    raw_text = reasoning or str(payload or "")
    return _extract_score(raw_text), trace


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
                    "expected": "non-empty in-character response",
                    "actual": actual_text,
                    "actual_raw": str(raw_actual or ""),
                    "passed": False,
                }
            ],
        }

    raw_context = item.get("raw_context")
    if not isinstance(raw_context, dict):
        raise ValueError("TimeChara items must expose raw_context.")
    prompt_template = _prompt_template("spatiotemporal_consistency_evaluation_prompt.txt")
    agent_name = str(raw_context.get("agent_name") or "").strip()
    question = str(raw_context.get("question") or "").strip()
    data_type = str(raw_context.get("data_type") or "").strip()
    temporal_label = str(raw_context.get("temporal_label") or "").strip()
    spatial_label = str(raw_context.get("spatial_label") or "").strip()

    trace_payload: dict[str, Any] = {}
    temporal_score: float | None = None
    spatial_score: float | None = None
    if data_type == "future" or data_type == "past-only":
        temporal_score, temporal_trace = _judge_score(
            task,
            prompt_template,
            agent_name=agent_name,
            question=question,
            answer=actual_text,
            fact=temporal_label,
            purpose=f"timechara_temporal:{item.get('item_id')}",
        )
        trace_payload["temporal"] = temporal_trace
    elif data_type in {"past-absence", "past-presence"}:
        spatial_score, spatial_trace = _judge_score(
            task,
            prompt_template,
            agent_name=agent_name,
            question=question,
            answer=actual_text,
            fact=spatial_label,
            purpose=f"timechara_spatial:{item.get('item_id')}",
        )
        trace_payload["spatial"] = spatial_trace
        if spatial_score and spatial_score > 0:
            temporal_score, temporal_trace = _judge_score(
                task,
                prompt_template,
                agent_name=agent_name,
                question=question,
                answer=actual_text,
                fact=temporal_label,
                purpose=f"timechara_temporal:{item.get('item_id')}",
            )
            trace_payload["temporal"] = temporal_trace
        else:
            temporal_score = 0.0
    else:
        raise ValueError(f"Unsupported TimeChara data_type={data_type!r}.")

    if temporal_score is None:
        objective = 0.0
        passed = False
    elif data_type in {"past-absence", "past-presence"}:
        objective = float(temporal_score) * float(spatial_score or 0.0)
        passed = True
    else:
        objective = float(temporal_score)
        passed = True

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return {
        "status": "pass" if passed else "fail",
        "verifier_status": "pass" if passed else "fail",
        "correctness": objective,
        "passed_tests": 1 if passed else 0,
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
                "expected": "__judge_model__",
                "actual": actual_text,
                "actual_raw": actual_text,
                "passed": passed,
                "temporal_score": temporal_score,
                "spatial_score": spatial_score,
            }
        ],
        "judge_trace": trace_payload or None,
    }
