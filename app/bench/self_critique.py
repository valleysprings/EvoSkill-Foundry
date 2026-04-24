from __future__ import annotations

import json
from typing import Any

_SELF_CRITIQUE_SYSTEM_PROMPT = (
    "You are a rigorous self-evaluator. Given a question and a model's answer, "
    "assess the confidence that the answer is correct — based solely on the quality "
    "of the reasoning and the internal consistency of the answer. "
    "Do not reveal or guess the ground truth. "
    "Return strict JSON with a single field: {\"confidence\": <float 0.0-1.0>}."
)

_SELF_CRITIQUE_THRESHOLD = 0.65


def self_critique_score(
    runtime: Any,
    *,
    task: dict[str, Any],
    candidate_output: str,
    question_item: dict[str, Any],
) -> float:
    """
    Black-box self-critique: ask the model to rate its own answer.
    Returns a confidence score in [0.0, 1.0].
    Ground truth is never exposed.
    """
    prompt_text = str(question_item.get("prompt") or "")
    choices = list(question_item.get("choices") or [])
    choices_text = ""
    if choices:
        choices_text = "\nChoices:\n" + "\n".join(f"  {i}. {c}" for i, c in enumerate(choices))

    user_prompt = (
        f"Question:\n{prompt_text}{choices_text}\n\n"
        f"Model answer:\n{candidate_output}\n\n"
        "How confident are you that this answer is correct? "
        "Consider: Is the reasoning sound and complete? Is the format appropriate? "
        "Is the answer internally consistent? "
        "Return JSON: {\"confidence\": <float 0.0-1.0>}"
    )
    try:
        payload, _trace = runtime.complete_json(
            purpose="self_critique",
            system_prompt=_SELF_CRITIQUE_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            queue_priority=10,
        )
        if isinstance(payload, dict):
            raw = payload.get("confidence")
            if raw is not None:
                return float(max(0.0, min(1.0, float(raw))))
    except Exception:  # noqa: BLE001
        pass
    return 0.0


def self_critique_outcome(confidence: float) -> str:
    """Map confidence score to 'success' / 'failure' outcome label."""
    return "success" if confidence >= _SELF_CRITIQUE_THRESHOLD else "failure"
