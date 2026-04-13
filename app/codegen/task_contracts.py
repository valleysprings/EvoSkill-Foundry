from __future__ import annotations

from typing import Any


TASK_MODE_VALUES = frozenset({"answer", "artifact"})
INTERACTION_MODE_VALUES = frozenset({"single_turn", "multi_turn"})
TASK_SHAPE_VALUES = frozenset({"mcq", "classification", "dialogue_next_turn", "dialogue_judgement", "agentic_open_ended"})
SCORING_MODE_VALUES = frozenset({"exact_match", "label_match", "rubric_score", "judge_model", "hybrid"})


def _normalized_string(value: Any) -> str:
    return str(value or "").strip()


def infer_task_mode(task: dict[str, Any]) -> str:
    explicit = _normalized_string(task.get("task_mode"))
    if not explicit:
        raise ValueError("Task must declare task_mode.")
    if explicit not in TASK_MODE_VALUES:
        raise ValueError(f"Invalid task_mode={explicit!r}; expected one of {sorted(TASK_MODE_VALUES)}.")
    return explicit


def infer_interaction_mode(task: dict[str, Any]) -> str:
    explicit = _normalized_string(task.get("interaction_mode"))
    if not explicit:
        raise ValueError("Task must declare interaction_mode.")
    if explicit not in INTERACTION_MODE_VALUES:
        raise ValueError(f"Invalid interaction_mode={explicit!r}; expected one of {sorted(INTERACTION_MODE_VALUES)}.")
    return explicit


def infer_task_shape(task: dict[str, Any]) -> str | None:
    explicit = _normalized_string(task.get("task_shape"))
    if not explicit:
        return None
    if explicit not in TASK_SHAPE_VALUES:
        raise ValueError(f"Invalid task_shape={explicit!r}; expected one of {sorted(TASK_SHAPE_VALUES)}.")
    return explicit


def infer_scoring_mode(task: dict[str, Any]) -> str | None:
    explicit = _normalized_string(task.get("scoring_mode"))
    if not explicit:
        return None
    if explicit not in SCORING_MODE_VALUES:
        raise ValueError(f"Invalid scoring_mode={explicit!r}; expected one of {sorted(SCORING_MODE_VALUES)}.")
    return explicit


def task_mode_summary(mode: str) -> str:
    if mode == "answer":
        return "Answer task: the candidate entrypoint returns the final answer or output for one evaluated item."
    if mode == "artifact":
        return "Artifact task: the returned or generated program, policy, script, or other artifact is what the verifier executes or consumes."
    return "Task contract summary unavailable."


def interaction_mode_summary(mode: str) -> str:
    if mode == "single_turn":
        return "Single-turn task: one candidate invocation returns the final result for each evaluated item."
    if mode == "multi_turn":
        return "Multi-turn task: candidate code acts over an episode with repeated observations, actions, and state updates."
    return "Interaction mode summary unavailable."
