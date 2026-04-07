from __future__ import annotations

import importlib
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


def _prompt_templates() -> dict[str, dict[str, str]]:
    source_root = ensure_repo_checkout("characterbench")
    prompt_root = source_root / "construct_prompts"
    prompt_root_text = str(prompt_root)
    if prompt_root_text not in sys.path:
        sys.path.insert(0, prompt_root_text)
    data_info_en = importlib.import_module("data_info_en")
    data_info = getattr(data_info_en, "data_info", None)
    if not isinstance(data_info, dict):
        raise ValueError("CharacterBench prompt templates are unavailable.")
    return data_info


def _dialogue_text(character_name: str, dialogue: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for index, turn in enumerate(dialogue):
        utterance = str(turn.get("utterance") or turn.get("text") or turn.get("content") or "").strip()
        if not utterance:
            continue
        speaker = "User" if index % 2 == 0 else character_name
        lines.append(f"{speaker}：{utterance}")
    return "\n".join(lines)


def _characterbench_prompt(item: dict[str, Any], response: str) -> str:
    raw_context = item.get("raw_context")
    if not isinstance(raw_context, dict):
        raise ValueError("CharacterBench items must expose raw_context.")
    data_info = _prompt_templates()
    subset = str(raw_context.get("subset") or "").strip()
    character_name = str(raw_context.get("character_name") or "").strip()
    character_profile = str(raw_context.get("character_profile") or "").strip()
    dialogue_history = _dialogue_text(character_name, [turn for turn in list(raw_context.get("dialogue") or []) if isinstance(turn, dict)])
    user_message = str(raw_context.get("user_message") or "").strip()
    candidate_response = f"{character_name}：{response}"
    user_line = f"User：{user_message}"
    if subset not in data_info:
        raise ValueError(f"Unsupported CharacterBench subset={subset!r}.")
    template = str((data_info.get(subset) or {}).get("base_eval_prompt") or "").strip()
    if not template:
        raise ValueError(f"Missing CharacterBench prompt template for subset={subset!r}.")
    if subset == "memory_consistency":
        reference_dialogue = "\n".join(f"{character_name}：{segment}" for segment in list(raw_context.get("reference_dialogue") or []))
        return template.format(
            character_profile=character_profile,
            dialogue="",
            user_query=user_line,
            character_response=candidate_response,
            reference=reference_dialogue,
        )
    if subset == "attribute_consistency_bot":
        return template.format(
            character_profile=character_profile,
            dialogue="",
            user_query=user_line,
            character_response=candidate_response,
            reference=str(raw_context.get("reference_answer") or "").strip(),
        )
    if subset == "attribute_consistency_human":
        return template.format(
            character_profile=character_profile,
            dialogue="",
            user_query=user_line,
            character_response=candidate_response,
            reference=str(raw_context.get("reference_answer") or "").strip(),
        )
    if subset == "behavior_consistency_bot":
        return template.format(
            character_profile=character_profile,
            dialogue=dialogue_history,
            user_query=user_line,
            character_response=candidate_response,
            reference=str(raw_context.get("reference_behavior") or "").strip(),
        )
    if subset == "behavior_consistency_human":
        return template.format(
            character_profile=character_profile + str(raw_context.get("reference_behavior") or "").strip(),
            dialogue=dialogue_history,
            reference=str(raw_context.get("reference_behavior") or "").strip(),
            user_query=user_line,
            character_response=candidate_response,
        )
    if subset == "boundary_consistency":
        return template.format(
            character_profile=character_profile,
            dialogue="",
            user_query=user_line,
            character_response=candidate_response,
            reference=str(raw_context.get("reference_response") or "").strip(),
        )
    if subset == "fact_accuracy":
        return template.format(
            character_profile=character_profile,
            dialogue="",
            user_query=user_line,
            character_response=candidate_response,
            reference=str(raw_context.get("reference_answer") or "").strip(),
        )
    if subset == "emotion_self_regulation":
        return template.format(
            character_profile=character_profile,
            dialogue="",
            user_query=user_line,
            character_response=candidate_response,
            reference=str(raw_context.get("reference_emotion") or "").strip(),
        )
    if subset == "empathetic_responsiveness":
        return template.format(
            character_profile=character_profile,
            dialogue="",
            user_query=user_line,
            character_response=candidate_response,
            reference=str(raw_context.get("reference_emotion") or "").strip(),
        )
    return template.format(
        character_profile=character_profile,
        dialogue=dialogue_history,
        user_query=user_line,
        character_response=candidate_response,
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
                    "expected": "non-empty character response",
                    "actual": actual_text,
                    "actual_raw": str(raw_actual or ""),
                    "passed": False,
                }
            ],
        }

    prompt = _characterbench_prompt(item, actual_text)
    payload, trace = run_eval_model_json(
        task=task,
        purpose=f"characterbench_judge:{item.get('item_id')}",
        system_prompt=(
            "You are running the CharacterBench evaluator. Use the supplied CharacterJudge prompt faithfully and "
            'return strict JSON only, for example {"score": 4, "summary": "..."} .'
        ),
        user_prompt=prompt,
    )
    raw_score = payload.get("score") if isinstance(payload, dict) else None
    try:
        score = float(raw_score)
    except (TypeError, ValueError):
        score = 0.0

    metadata = dict(item.get("metadata") or {})
    subset_max = float(metadata.get("subset_max_annotation_score") or 5.0)
    objective = max(0.0, min(1.0, round(score / max(subset_max, 1.0), 6)))
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
                "expected": "__judge_model__",
                "actual": actual_text,
                "actual_raw": actual_text,
                "passed": objective > 0.0,
                "judge_score": score,
            }
        ],
        "judge_trace": trace,
    }
