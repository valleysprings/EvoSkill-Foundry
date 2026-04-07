from __future__ import annotations

import ast
import json
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


def _load_questionnaire(name: str) -> dict[str, Any]:
    source_root = ensure_repo_checkout("incharacter")
    path = source_root / "data" / "questionnaires" / f"{name}.json"
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Expected questionnaire metadata object at {path}.")
    return payload


def _parse_answers(raw_actual: object) -> dict[str, str] | None:
    if isinstance(raw_actual, dict):
        candidate = raw_actual
    else:
        text = str(raw_actual or "").strip()
        if not text:
            return None
        try:
            candidate = json.loads(text)
        except json.JSONDecodeError:
            try:
                candidate = ast.literal_eval(text)
            except (SyntaxError, ValueError):
                return None
    if not isinstance(candidate, dict):
        return None
    parsed: dict[str, str] = {}
    for key, value in candidate.items():
        question_id = str(key or "").strip()
        answer = str(value or "").strip()
        if question_id and answer:
            parsed[question_id] = answer
    return parsed or None


def _official_prompt(questionnaire: dict[str, Any], dimension: str, character_name: str) -> str:
    prompts = questionnaire.get("prompts")
    if not isinstance(prompts, dict):
        raise ValueError("InCharacter questionnaire metadata must expose prompts.")
    dimension_prompt = prompts.get("convert_to_choice_adjoption")
    dimension_instruction = prompts.get("llm_choice_instruction_adjoption")
    if isinstance(dimension_prompt, dict) and isinstance(dimension_instruction, dict):
        prompt_entry = dimension_prompt.get(dimension)
        instruction_entry = dimension_instruction.get(dimension)
        if isinstance(prompt_entry, dict) and isinstance(instruction_entry, dict):
            prompt_text = str(prompt_entry.get("en") or "").strip()
            instruction_text = str(instruction_entry.get("en") or "").strip()
            if prompt_text and instruction_text:
                return f"{prompt_text}\n{instruction_text}".replace("<character>", character_name)
    convert_prompt = str((prompts.get("convert_to_choice") or {}).get("en") or "").strip()
    choice_instruction = str((prompts.get("llm_choice_instruction") or {}).get("en") or "").strip()
    if not convert_prompt or not choice_instruction:
        raise ValueError("InCharacter questionnaire metadata is missing the official conversion prompts.")
    return f"{convert_prompt}\n{choice_instruction}".replace("<character>", character_name)


def _replace_all(text: str, replacements: list[tuple[str, str]]) -> str:
    updated = text
    for source, target in replacements:
        if source:
            updated = updated.replace(source, target)
    return updated


def _anonymize_conversion_inputs(
    *,
    system_prompt: str,
    payload: dict[str, Any],
    aliases: list[str],
    experimenter: str,
) -> tuple[str, dict[str, Any]]:
    replacements = [(alias, "<the participant>") for alias in sorted(aliases, key=len, reverse=True) if alias]
    if experimenter:
        replacements.append((experimenter, "<the experimenter>"))
    anonymized_system_prompt = _replace_all(system_prompt, replacements)
    anonymized_payload = json.loads(json.dumps(payload, ensure_ascii=False))
    for question_id, entry in anonymized_payload.items():
        if not isinstance(entry, dict):
            continue
        for field in ("statement", "conversation"):
            entry[field] = _replace_all(str(entry.get(field) or ""), replacements)
    return anonymized_system_prompt, anonymized_payload


def _normalized_score(
    *,
    questionnaire_name: str,
    questionnaire_range: list[object],
    questions: list[dict[str, Any]],
    converted_choices: dict[str, object],
    dimension: str,
) -> tuple[float | None, dict[str, float]]:
    try:
        low = float(questionnaire_range[0])
        high = float(questionnaire_range[1])
    except (IndexError, TypeError, ValueError):
        raise ValueError("InCharacter item metadata must include questionnaire_range.")
    midpoint = (low + high) / 2.0
    converted: dict[str, float] = {}
    scores: list[float] = []
    for question in questions:
        question_id = str(question.get("id") or "").strip()
        if not question_id:
            continue
        raw_choice = converted_choices.get(question_id)
        if raw_choice is None and question_id.isdigit():
            raw_choice = converted_choices.get(str(int(question_id)))
        if raw_choice is None:
            return None, {}
        text_choice = str(raw_choice).strip().lower()
        if text_choice == "x":
            choice = midpoint
        else:
            try:
                choice = float(raw_choice)
            except (TypeError, ValueError):
                return None, {}
        category = str(question.get("category") or "").strip()
        if questionnaire_name == "16Personalities":
            positive = bool(dimension) and category == dimension[0]
        else:
            positive = category != "negative"
        score = choice if positive else (low + high - choice)
        converted[question_id] = choice
        if questionnaire_name == "16Personalities":
            score = ((score - low) / (high - low)) * 100.0
        scores.append(score)
    if not scores:
        return None, {}
    return sum(scores) / len(scores), converted


def _target_score(questionnaire_name: str, annotation_score: object, questionnaire_range: list[object]) -> float:
    text_score = str(annotation_score or "").strip()
    if text_score and text_score.upper() != "X":
        try:
            return float(text_score)
        except ValueError:
            pass
    if questionnaire_name == "16Personalities":
        return 50.0
    try:
        low = float(questionnaire_range[0])
        high = float(questionnaire_range[1])
    except (IndexError, TypeError, ValueError):
        raise ValueError("InCharacter item metadata must include questionnaire_range.")
    return (low + high) / 2.0


def _label_from_score(questionnaire_name: str, score: float, questionnaire_range: list[object]) -> str:
    if questionnaire_name == "16Personalities":
        span = 100.0
        threshold = 50.0
    else:
        low = float(questionnaire_range[0])
        high = float(questionnaire_range[1])
        span = high - low
        threshold = (low + high) / 2.0
    neutral_band = max(span * 0.1, 0.25)
    if score > threshold + neutral_band:
        return "H"
    if score < threshold - neutral_band:
        return "L"
    return "X"


def _dimension_conversion_payload(
    *,
    questions: list[dict[str, Any]],
    answers: dict[str, str],
    character_name: str,
) -> dict[str, dict[str, str]]:
    return {
        question_id: {
            "statement": str(question.get("statement") or "").strip(),
            "conversation": f'Experimenter: {str(question.get("question") or "").strip()}\n{character_name}: {answers[question_id]}',
        }
        for question in questions
        if isinstance(question, dict)
        for question_id in [str(question.get("id") or "").strip()]
        if question_id and question_id in answers
    }


def evaluate_candidate(*, task, candidate_path, source_code, baseline_metrics, memory_applied):
    del source_code, baseline_metrics, memory_applied
    item = task.get("question_item")
    if not isinstance(item, dict):
        raise ValueError("Dataset question task must provide question_item.")

    started = time.perf_counter()
    solver = load_callable_from_path(Path(candidate_path), str(task["entry_symbol"]))
    raw_actual = solver(public_question_payload(item))
    answers = _parse_answers(raw_actual)
    if answers is None:
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
                    "expected": "strict JSON object mapping question ids to answer strings",
                    "actual": str(raw_actual or ""),
                    "actual_raw": str(raw_actual or ""),
                    "passed": False,
                }
            ],
        }

    raw_context = item.get("raw_context")
    if not isinstance(raw_context, dict):
        raise ValueError("InCharacter items must provide raw_context.")
    questions = list(raw_context.get("questions") or [])
    question_ids = [str(question.get("id") or "").strip() for question in questions if isinstance(question, dict)]
    if not question_ids:
        raise ValueError("InCharacter items must provide questionnaire questions.")
    if any(question_id not in answers for question_id in question_ids):
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
                    "expected": question_ids,
                    "actual": sorted(answers),
                    "actual_raw": str(raw_actual or ""),
                    "passed": False,
                }
            ],
        }

    metadata = dict(item.get("metadata") or {})
    questionnaire_name = str(metadata.get("questionnaire") or "").strip()
    character_name = str(raw_context.get("character") or "").strip()
    aliases = [str(alias).strip() for alias in list(raw_context.get("aliases") or []) if str(alias).strip()]
    experimenter = str(raw_context.get("experimenter") or "").strip()
    questionnaire = _load_questionnaire(questionnaire_name)
    question_by_id = {
        str(question.get("id") or "").strip(): question
        for question in questions
        if isinstance(question, dict) and str(question.get("id") or "").strip()
    }
    dimension_questions = {
        str(dimension).strip(): [str(question_id).strip() for question_id in list(question_ids) if str(question_id).strip()]
        for dimension, question_ids in dict(raw_context.get("dimension_questions") or {}).items()
        if str(dimension).strip()
    }
    annotation_labels = {
        str(dimension).strip(): value
        for dimension, value in dict(metadata.get("annotation_labels") or {}).items()
        if str(dimension).strip()
    }
    pdb_labels = {
        str(dimension).strip(): value
        for dimension, value in dict(metadata.get("pdb_labels") or {}).items()
        if str(dimension).strip()
    }
    if not dimension_questions or not annotation_labels or not pdb_labels:
        raise ValueError("InCharacter items must provide dimension-level labels and question groupings.")

    dimension_results: dict[str, Any] = {}
    judge_trace: dict[str, Any] = {}
    valid_dimensions = 0
    correct_dimensions = 0
    full_profile_correct = True
    for dimension in sorted(annotation_labels):
        annotation = annotation_labels.get(dimension)
        pdb_label = dict(pdb_labels.get(dimension) or {})
        if not isinstance(annotation, dict):
            continue
        grouped_questions = [question_by_id[question_id] for question_id in dimension_questions.get(dimension, []) if question_id in question_by_id]
        if not grouped_questions:
            raise ValueError(f"InCharacter item is missing question rows for dimension={dimension!r}.")
        system_prompt = _official_prompt(questionnaire, dimension, character_name)
        conversion_payload = _dimension_conversion_payload(
            questions=grouped_questions,
            answers=answers,
            character_name=character_name,
        )
        system_prompt, conversion_payload = _anonymize_conversion_inputs(
            system_prompt=system_prompt,
            payload=conversion_payload,
            aliases=aliases,
            experimenter=experimenter,
        )
        converted_choices, trace = run_eval_model_json(
            task=task,
            purpose=f"incharacter_choice_conversion:{item.get('item_id')}:{dimension}",
            system_prompt=system_prompt,
            user_prompt=json.dumps(conversion_payload, indent=2, ensure_ascii=False),
        )
        if not isinstance(converted_choices, dict):
            converted_choices = {}
        score, normalized_choices = _normalized_score(
            questionnaire_name=questionnaire_name,
            questionnaire_range=list(metadata.get("questionnaire_range") or []),
            questions=grouped_questions,
            converted_choices=converted_choices,
            dimension=dimension,
        )
        expected_label = str(annotation.get("type") or "").strip().upper()
        if score is None:
            predicted_label = None
            label_match = False
            dimension_score = None
        else:
            predicted_label = _label_from_score(questionnaire_name, score, list(metadata.get("questionnaire_range") or []))
            label_match = predicted_label == expected_label
            dimension_score = round(score, 6)
        is_valid_dimension = str(pdb_label.get("type") or "").strip().upper() != "X"
        if is_valid_dimension:
            valid_dimensions += 1
            if score is None:
                full_profile_correct = False
            elif label_match:
                correct_dimensions += 1
            else:
                full_profile_correct = False
        dimension_results[dimension] = {
            "expected_label": expected_label,
            "predicted_label": predicted_label,
            "valid_for_metrics": is_valid_dimension,
            "label_match": label_match,
            "dimension_score": dimension_score,
            "target_score": _target_score(questionnaire_name, annotation.get("score"), list(metadata.get("questionnaire_range") or [])),
            "converted_choices": normalized_choices or None,
        }
        judge_trace[dimension] = trace

    if valid_dimensions <= 0:
        raise ValueError("InCharacter item must include at least one valid metric dimension.")
    objective = round(correct_dimensions / valid_dimensions, 6)
    predicted_labels = {dimension: result["predicted_label"] for dimension, result in dimension_results.items()}
    expected_labels = {dimension: result["expected_label"] for dimension, result in dimension_results.items()}

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
                "expected": expected_labels,
                "actual": predicted_labels,
                "actual_raw": json.dumps(answers, ensure_ascii=False, sort_keys=True),
                "passed": full_profile_correct,
                "single_dimension_accuracy": objective,
                "full_profile_accuracy": 1.0 if full_profile_correct else 0.0,
                "correct_dimensions": correct_dimensions,
                "valid_dimensions": valid_dimensions,
                "dimension_results": dimension_results,
            }
        ],
        "judge_trace": judge_trace,
    }
