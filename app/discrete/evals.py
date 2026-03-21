from __future__ import annotations

import math
from typing import Any


def quality_metrics(direction: str, baseline_objective: float, objective: float) -> tuple[float, float]:
    if baseline_objective <= 0:
        return 0.0, 0.0
    if direction == "min":
        relative_gain = max(0.0, (baseline_objective - objective) / baseline_objective)
    else:
        relative_gain = max(0.0, (objective - baseline_objective) / baseline_objective)
    quality_score = min(relative_gain / 0.35, 1.0)
    return relative_gain, quality_score


def score_candidate(
    *,
    direction: str,
    objective: float,
    baseline_objective: float,
    feasible: bool,
    latency_ms: float,
    complexity: float,
    memory_applied: bool,
) -> dict[str, Any]:
    relative_gain, quality_score = quality_metrics(direction, baseline_objective, objective)
    stability = 1.0 if feasible else 0.0
    latency_norm = min(latency_ms / 4.0, 1.0)
    memory_bonus = 1.0 if memory_applied else 0.0
    score = (
        1.25 * float(feasible)
        + 1.20 * quality_score
        + 0.25 * memory_bonus
        + 0.15 * stability
        - 0.12 * latency_norm
        - 0.10 * complexity
    )
    return {
        "feasible": feasible,
        "objective": round(objective, 3),
        "relative_gain": round(relative_gain, 4),
        "quality_score": round(quality_score, 4),
        "latency_ms": round(latency_ms, 3),
        "stability": round(stability, 3),
        "complexity": round(complexity, 2),
        "memory_bonus": round(memory_bonus, 3),
        "J": round(score, 4),
    }


def is_better(direction: str, left: float, right: float) -> bool:
    return left < right if direction == "min" else left > right


def _distance(left: dict[str, float], right: dict[str, float]) -> float:
    return math.hypot(left["x"] - right["x"], left["y"] - right["y"])


def _tsp_length(route: list[int], nodes: list[dict[str, float]]) -> float:
    total = 0.0
    for index, node_id in enumerate(route):
        left = nodes[node_id]
        right = nodes[route[(index + 1) % len(route)]]
        total += _distance(left, right)
    return total


def _cut_weight(assignment: list[int], edges: list[tuple[int, int, int]]) -> float:
    return float(sum(weight for left, right, weight in edges if assignment[left] != assignment[right]))


def route_visual(task: dict[str, Any], route: list[int]) -> dict[str, Any]:
    return {
        "type": "tsp",
        "nodes": task["instance"]["nodes"],
        "route": route,
        "objective": round(_tsp_length(route, task["instance"]["nodes"]), 3),
    }


def max_cut_visual(task: dict[str, Any], assignment: list[int]) -> dict[str, Any]:
    return {
        "type": "max-cut",
        "nodes": task["instance"]["nodes"],
        "edges": [
            {"left": left, "right": right, "weight": weight}
            for left, right, weight in task["instance"]["edges"]
        ],
        "assignment": assignment,
        "objective": round(_cut_weight(assignment, task["instance"]["edges"]), 3),
    }


def solution_visual(task: dict[str, Any], solution: list[int]) -> dict[str, Any]:
    if task["task_kind"] == "tsp":
        return route_visual(task, solution)
    return max_cut_visual(task, solution)


def candidate_summary(task: dict[str, Any], candidate: dict[str, Any]) -> str:
    if task["task_kind"] == "tsp":
        preview = " -> ".join(str(node_id) for node_id in candidate["solution"][:6])
        return f"{preview} -> ..."
    left = [str(index) for index, side in enumerate(candidate["solution"]) if side == 0]
    right = [str(index) for index, side in enumerate(candidate["solution"]) if side == 1]
    return f"L[{', '.join(left[:4])}] | R[{', '.join(right[:4])}]"
