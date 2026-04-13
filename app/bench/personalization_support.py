from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

from app.bench.benchmark_support import (
    choice_answer_matches,
    choice_response_display,
    normalize_answer_text,
    preview_display_text,
    public_question_payload,
)
from app.codegen.errors import ConfigError
from app.codegen.llm import ProposalRuntime
from app.codegen.verifier import load_callable_from_path


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True))
    temp_path.replace(path)


def _dedupe_strings(values: Iterable[object]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        key = normalize_answer_text(text)
        if not key or key in seen:
            continue
        normalized.append(text)
        seen.add(key)
    return normalized


def _best_label_alias(label: str, label_aliases: dict[str, Sequence[object]] | None = None) -> str | None:
    normalized_label = normalize_answer_text(label)
    for alias in _dedupe_strings(dict(label_aliases or {}).get(label) or []):
        normalized_alias = normalize_answer_text(alias)
        if not normalized_alias or normalized_alias == normalized_label:
            continue
        if normalized_alias in {"a", "b", "c", "d"}:
            continue
        return alias
    return None


def label_response_display(
    raw_actual: object,
    *,
    actual_label: str | None,
    label_aliases: dict[str, Sequence[object]] | None = None,
) -> str | None:
    if actual_label:
        alias = _best_label_alias(actual_label, label_aliases)
        if alias:
            return f"{actual_label} -> {preview_display_text(alias) or alias}"
        return actual_label
    raw_text = str(raw_actual or "").strip()
    if not raw_text:
        return None
    return preview_display_text(raw_text)


def benchmark_metadata(
    *,
    benchmark: str,
    benchmark_category: str,
    interaction_mode: str,
    task_shape: str,
    scoring_mode: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = {
        "benchmark": benchmark,
        "benchmark_category": benchmark_category,
        "interaction_mode": interaction_mode,
        "task_shape": task_shape,
        "scoring_mode": scoring_mode,
    }
    if isinstance(extra, dict):
        metadata.update(extra)
    return metadata


def serialize_dialogue_history(history: Sequence[dict[str, Any] | tuple[str, str]]) -> str:
    lines: list[str] = []
    for turn in history:
        if isinstance(turn, tuple) and len(turn) == 2:
            speaker, text = turn
        elif isinstance(turn, dict):
            speaker = turn.get("speaker") or turn.get("role") or turn.get("from") or "Speaker"
            text = turn.get("text") or turn.get("content") or turn.get("value") or ""
        else:
            continue
        speaker_text = str(speaker or "").strip() or "Speaker"
        message_text = str(text or "").strip()
        if not message_text:
            continue
        lines.append(f"{speaker_text}: {message_text}")
    return "\n\n".join(lines)


_SOCIALBENCH_SECTION_PATTERN = re.compile(r"(?m)^==\s*([^=\n]+?)\s*==\s*$")
_SOCIALBENCH_DIALOGUE_LINE_PATTERN = re.compile(r"^\s*([^:\n]+):\s*(.+?)\s*$")
_SOCIALBENCH_OPTION_LINE_PATTERN = re.compile(r"^\s*([A-H])\.\s*(.+?)\s*$")


def _socialbench_sections(text: str) -> dict[str, str]:
    matches = list(_SOCIALBENCH_SECTION_PATTERN.finditer(text))
    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        sections[match.group(1).strip()] = text[start:end].strip()
    return sections


def _socialbench_split_dialogue_block(block: str) -> tuple[list[str], list[str]]:
    dialogue_lines: list[str] = []
    tail_lines: list[str] = []
    seen_dialogue = False
    lines = block.splitlines()
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        is_option = _SOCIALBENCH_OPTION_LINE_PATTERN.match(stripped) is not None
        is_dialogue = _SOCIALBENCH_DIALOGUE_LINE_PATTERN.match(stripped) is not None and not is_option
        if is_dialogue and not tail_lines:
            dialogue_lines.append(stripped)
            seen_dialogue = True
            continue
        if seen_dialogue:
            tail_lines = [line, *lines[index + 1 :]]
            break
        tail_lines = lines[index:]
        break
    return dialogue_lines, tail_lines


def _socialbench_dialogue_history(lines: Sequence[str]) -> list[dict[str, str]]:
    history: list[dict[str, str]] = []
    for line in lines:
        match = _SOCIALBENCH_DIALOGUE_LINE_PATTERN.match(line.strip())
        if match is None:
            continue
        history.append({"speaker": match.group(1).strip(), "text": match.group(2).strip()})
    return history


def _socialbench_options(lines: Sequence[str]) -> list[dict[str, str]]:
    parsed: list[dict[str, str]] = []
    for line in lines:
        match = _SOCIALBENCH_OPTION_LINE_PATTERN.match(line.strip())
        if match is None:
            continue
        parsed.append({"label": match.group(1), "text": match.group(2).strip()})
    return parsed


def _socialbench_instruction(lines: Sequence[str]) -> str:
    fragments: list[str] = []
    for raw_line in lines:
        line = raw_line.strip()
        if not line or _SOCIALBENCH_OPTION_LINE_PATTERN.match(line):
            continue
        normalized = line.rstrip(":：").strip().lower()
        if normalized in {"your selection", "你的选择", "assistant"}:
            continue
        fragments.append(line)
    return " ".join(fragments).strip()


def _socialbench_latest_user_message(history: Sequence[dict[str, str]]) -> str:
    for turn in reversed(history):
        speaker = str(turn.get("speaker") or "").strip().lower()
        if speaker in {"user", "用户"}:
            return str(turn.get("text") or "").strip()
    return ""


def parse_socialbench_prompt(prompt: object, *, metadata: dict[str, Any] | None = None) -> dict[str, Any] | None:
    text = str(prompt or "").strip()
    if not text:
        return None

    metadata_map = dict(metadata or {})
    sections = _socialbench_sections(text)
    conversation_block = sections.get("Conversations") or sections.get("对话历史") or text
    dialogue_lines, tail_lines = _socialbench_split_dialogue_block(conversation_block)
    history = _socialbench_dialogue_history(dialogue_lines)
    options = _socialbench_options(tail_lines)
    instruction = _socialbench_instruction(tail_lines)

    target_utterance = ""
    utterance_match = re.search(r'utterance\s+"([^"]+)"', instruction, flags=re.IGNORECASE)
    if utterance_match is None:
        utterance_match = re.search(r'最符合"([^"]+)"说话者', instruction)
    if utterance_match is not None:
        target_utterance = utterance_match.group(1).strip()

    latest_user_message = _socialbench_latest_user_message(history)
    question_text = target_utterance or latest_user_message or instruction
    role_name = str(metadata_map.get("role_name") or metadata_map.get("name") or "").strip()

    payload = {
        "benchmark": "socialbench",
        "category": str(metadata_map.get("category") or "").strip(),
        "lang": str(metadata_map.get("lang") or "").strip(),
        "role_name": role_name,
        "profile": sections.get("Profile") or sections.get("角色描述") or "",
        "profiles": sections.get("Profiles") or "",
        "dialogue_history": history,
        "latest_user_message": latest_user_message,
        "target_utterance": target_utterance,
        "instruction": instruction,
        "question_text": question_text,
        "options": options,
    }
    return {key: value for key, value in payload.items() if value not in ("", [], None)}


def format_rubric_input(item: dict[str, Any], response: object) -> str:
    prompt = str(item.get("prompt") or "").strip()
    context = item.get("raw_context") if item.get("raw_context") is not None else item.get("context")
    history_text = ""
    if isinstance(context, dict) and isinstance(context.get("dialogue_history"), list):
        history_text = serialize_dialogue_history(list(context.get("dialogue_history") or []))
    elif isinstance(context, list):
        history_text = serialize_dialogue_history(context)
    else:
        history_text = str(context or "").strip()
    return "\n\n".join(
        block
        for block in (
            f"Prompt:\n{prompt}" if prompt else "",
            f"Context:\n{history_text}" if history_text else "",
            f"Candidate Response:\n{str(response or '').strip()}",
        )
        if block
    )


def make_choice_item(
    *,
    item_id: str,
    name: str,
    prompt: str,
    choices: Sequence[str],
    correct_choice_index: int,
    context: str | None = None,
    raw_context: Any = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    normalized_choices = [str(choice).strip() for choice in choices]
    expected_answer = normalized_choices[correct_choice_index]
    choice_letter = chr(ord("A") + correct_choice_index)
    merged_metadata = dict(metadata or {})
    merged_metadata["correct_choice_index"] = correct_choice_index
    merged_metadata["answer_aliases"] = _dedupe_strings(
        [choice_letter, expected_answer, *(merged_metadata.get("answer_aliases") or [])]
    )
    item = {
        "item_id": item_id,
        "name": name,
        "prompt": str(prompt).strip(),
        "choices": normalized_choices,
        "expected_answer": expected_answer,
        "metadata": merged_metadata,
    }
    if context:
        item["context"] = str(context).strip()
    if raw_context is not None:
        item["raw_context"] = raw_context
    return item


def make_label_item(
    *,
    item_id: str,
    name: str,
    prompt: str,
    expected_label: str,
    allowed_labels: Sequence[str],
    context: str | None = None,
    raw_context: Any = None,
    metadata: dict[str, Any] | None = None,
    label_aliases: dict[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    merged_metadata = dict(metadata or {})
    normalized_allowed = [str(label).strip() for label in allowed_labels if str(label).strip()]
    merged_metadata["allowed_labels"] = normalized_allowed
    alias_payload = {
        str(label).strip(): _dedupe_strings(values)
        for label, values in dict(label_aliases or {}).items()
        if str(label).strip()
    }
    if alias_payload:
        merged_metadata["label_aliases"] = alias_payload
    item = {
        "item_id": item_id,
        "name": name,
        "prompt": str(prompt).strip(),
        "expected_answer": str(expected_label).strip(),
        "metadata": merged_metadata,
    }
    if context:
        item["context"] = str(context).strip()
    if raw_context is not None:
        item["raw_context"] = raw_context
    return item


def make_dialogue_next_turn_item(
    *,
    item_id: str,
    name: str,
    prompt: str,
    dialogue_history: Sequence[dict[str, Any] | tuple[str, str]],
    expected_reply: str,
    metadata: dict[str, Any] | None = None,
    response_aliases: Sequence[str] | None = None,
) -> dict[str, Any]:
    merged_metadata = dict(metadata or {})
    merged_metadata["response_aliases"] = _dedupe_strings([expected_reply, *(response_aliases or [])])
    raw_history: list[dict[str, str]] = []
    for turn in dialogue_history:
        if isinstance(turn, tuple) and len(turn) == 2:
            speaker, text = turn
        elif isinstance(turn, dict):
            speaker = turn.get("speaker") or turn.get("role") or turn.get("from") or "Speaker"
            text = turn.get("text") or turn.get("content") or turn.get("value") or ""
        else:
            continue
        raw_history.append({"speaker": str(speaker or "").strip() or "Speaker", "text": str(text or "").strip()})
    return {
        "item_id": item_id,
        "name": name,
        "prompt": str(prompt).strip(),
        "context": serialize_dialogue_history(raw_history),
        "raw_context": {"dialogue_history": raw_history},
        "expected_answer": str(expected_reply).strip(),
        "metadata": merged_metadata,
    }


def write_manifest(
    path: Path,
    *,
    dataset_id: str,
    split: str,
    items: Sequence[dict[str, Any]],
    dataset_size: int | None = None,
) -> dict[str, Any]:
    manifest = {
        "dataset_id": str(dataset_id).strip(),
        "split": str(split).strip(),
        "dataset_size": int(dataset_size if dataset_size is not None else len(items)),
        "prepared_count": len(items),
        "items": [dict(item) for item in items],
    }
    _write_json(path, manifest)
    return manifest


def eval_model_name(task: dict[str, Any]) -> str | None:
    value = str(task.get("eval_model") or "").strip()
    return value or None


def run_eval_model_json(
    *,
    task: dict[str, Any],
    purpose: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    eval_model = eval_model_name(task)
    if not eval_model:
        raise ConfigError("This personalization verifier requires eval_model.")
    runtime = ProposalRuntime.from_env().with_model(eval_model)
    return runtime.complete_json(
        purpose=purpose,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        queue_priority=1500,
    )


def clamp_score(value: object, *, minimum: float = 0.0, maximum: float = 1.0) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return max(minimum, min(maximum, numeric))


def mean_score(values: Sequence[object], *, minimum: float = 0.0, maximum: float = 1.0) -> float | None:
    normalized: list[float] = []
    for value in values:
        score = clamp_score(value, minimum=minimum, maximum=maximum)
        if score is not None:
            normalized.append(score)
    if not normalized:
        return None
    return sum(normalized) / len(normalized)


def parse_label_prediction(
    actual: object,
    *,
    allowed_labels: Sequence[object],
    label_aliases: dict[str, Sequence[object]] | None = None,
) -> str | None:
    actual_text = normalize_answer_text(actual)
    if not actual_text:
        return None
    for label in allowed_labels:
        normalized_label = str(label).strip()
        if not normalized_label:
            continue
        aliases = _dedupe_strings([normalized_label, *(dict(label_aliases or {}).get(normalized_label) or [])])
        if actual_text in {normalize_answer_text(alias) for alias in aliases}:
            return normalized_label
    for label in allowed_labels:
        normalized_label = str(label).strip()
        if not normalized_label:
            continue
        if normalize_answer_text(normalized_label) in actual_text:
            return normalized_label
    return None


def _question_item(task: dict[str, Any]) -> dict[str, Any]:
    item = task.get("question_item")
    if not isinstance(item, dict):
        raise ValueError("Dataset question task must provide question_item.")
    return item


def _solver_output(task: dict[str, Any], candidate_path: Path, item: dict[str, Any]) -> tuple[object, float]:
    started = time.perf_counter()
    solver = load_callable_from_path(candidate_path, str(task["entry_symbol"]))
    raw_actual = solver(public_question_payload(item))
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return raw_actual, elapsed_ms


def _verifier_payload(
    *,
    item: dict[str, Any],
    actual: str,
    actual_raw: object,
    actual_display: str | None = None,
    passed: bool,
    objective: float,
    elapsed_ms: float,
    well_formed: bool = True,
) -> dict[str, Any]:
    row = {
        "name": item.get("name") or item["item_id"],
        "expected": item.get("expected_answer"),
        "actual": actual,
        "actual_raw": str(actual_raw or ""),
        "passed": passed,
    }
    if actual_display:
        row["actual_display"] = actual_display
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
        "test_results": [row],
    }


def evaluate_choice_candidate(
    *,
    task: dict[str, Any],
    candidate_path: Path,
) -> dict[str, Any]:
    item = _question_item(task)
    raw_actual, elapsed_ms = _solver_output(task, candidate_path, item)
    passed, actual = choice_answer_matches(
        raw_actual,
        expected=item["expected_answer"],
        choices=item.get("choices") or [],
        answer_alias_list=item.get("metadata", {}).get("answer_aliases", []),
        correct_choice_index=item.get("metadata", {}).get("correct_choice_index"),
    )
    objective = 1.0 if passed else 0.0
    actual_display = choice_response_display(
        actual,
        raw_actual=raw_actual,
        choices=item.get("choices") or [],
        preferred_choice_index=item.get("metadata", {}).get("correct_choice_index") if passed else None,
    )
    return _verifier_payload(
        item=item,
        actual=actual,
        actual_raw=raw_actual,
        actual_display=actual_display,
        passed=passed,
        objective=objective,
        elapsed_ms=elapsed_ms,
        well_formed=bool(str(raw_actual or "").strip()),
    )


def evaluate_label_candidate(
    *,
    task: dict[str, Any],
    candidate_path: Path,
) -> dict[str, Any]:
    item = _question_item(task)
    raw_actual, elapsed_ms = _solver_output(task, candidate_path, item)
    metadata = dict(item.get("metadata") or {})
    actual_label = parse_label_prediction(
        raw_actual,
        allowed_labels=list(metadata.get("allowed_labels") or []),
        label_aliases=dict(metadata.get("label_aliases") or {}),
    )
    expected_label = str(item.get("expected_answer") or "").strip()
    passed = actual_label == expected_label and actual_label is not None
    objective = 1.0 if passed else 0.0
    actual = actual_label or normalize_answer_text(raw_actual)
    actual_display = label_response_display(
        raw_actual,
        actual_label=actual_label,
        label_aliases=dict(metadata.get("label_aliases") or {}),
    )
    return _verifier_payload(
        item=item,
        actual=actual,
        actual_raw=raw_actual,
        actual_display=actual_display,
        passed=passed,
        objective=objective,
        elapsed_ms=elapsed_ms,
        well_formed=actual_label is not None,
    )


def evaluate_exact_text_candidate(
    *,
    task: dict[str, Any],
    candidate_path: Path,
) -> dict[str, Any]:
    item = _question_item(task)
    raw_actual, elapsed_ms = _solver_output(task, candidate_path, item)
    actual_text = str(raw_actual or "").strip()
    aliases = _dedupe_strings([item.get("expected_answer"), *(item.get("metadata", {}).get("response_aliases") or [])])
    passed = normalize_answer_text(actual_text) in {normalize_answer_text(alias) for alias in aliases if str(alias).strip()}
    objective = 1.0 if passed else 0.0
    return _verifier_payload(
        item=item,
        actual=actual_text,
        actual_raw=raw_actual,
        actual_display=actual_text,
        passed=passed,
        objective=objective,
        elapsed_ms=elapsed_ms,
        well_formed=bool(actual_text),
    )
