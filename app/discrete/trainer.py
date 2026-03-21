from __future__ import annotations

import time
from typing import Any, Callable

from app.discrete.evals import candidate_summary, is_better, score_candidate, solution_visual
from app.discrete.operations import cut_weight, max_cut_specs, solve_candidate, tsp_length, tsp_specs
from app.memory.store import MemoryStore

ProgressCallback = Callable[[dict[str, Any]], None]


def _emit(progress_callback: ProgressCallback | None, pace_ms: int, **payload: Any) -> None:
    if progress_callback is not None:
        progress_callback(payload)
    if pace_ms > 0:
        time.sleep(pace_ms / 1000.0)


def _archive_sort_key(direction: str, candidate: dict[str, Any]) -> tuple[float, float]:
    objective = candidate["metrics"]["objective"]
    signed_objective = -objective if direction == "min" else objective
    return signed_objective, candidate["metrics"]["J"]


def _update_archive(direction: str, archive: list[dict[str, Any]], candidate: dict[str, Any]) -> list[dict[str, Any]]:
    signature = tuple(candidate["solution"])
    deduped = [item for item in archive if tuple(item["solution"]) != signature]
    deduped.append(candidate)
    deduped.sort(key=lambda item: _archive_sort_key(direction, item), reverse=True)
    return deduped[:4]


def _baseline_candidate(task: dict[str, Any]) -> dict[str, Any]:
    if task["task_kind"] == "tsp":
        solution = list(range(len(task["instance"]["nodes"])))
        objective = tsp_length(solution, task["instance"]["nodes"])
    else:
        solution = [index % 2 for index in range(len(task["instance"]["nodes"]))]
        objective = cut_weight(solution, task["instance"]["edges"])

    metrics = score_candidate(
        direction=task["objective_direction"],
        objective=objective,
        baseline_objective=objective,
        feasible=True,
        latency_ms=0.0,
        complexity=0.18,
        memory_applied=False,
    )
    baseline = {
        "candidate_id": f"{task['id']}-baseline",
        "agent": "baseline",
        "label": "Initial heuristic",
        "operator_family": "baseline",
        "strategy": "Use the naive checked-in solution with no replay and no improvement loop.",
        "solution": solution,
        "supporting_memory_ids": [],
        "reusable_rules": [],
        "metrics": metrics,
        "parent_ids": [],
    }
    baseline["visual"] = solution_visual(task, solution)
    baseline["solution_summary"] = candidate_summary(task, baseline)
    return baseline


def _build_experience(
    task: dict[str, Any],
    generation: int,
    previous_best: dict[str, Any],
    new_best: dict[str, Any],
    delta_j: float,
) -> dict[str, Any]:
    previous_objective = previous_best["metrics"]["objective"]
    new_objective = new_best["metrics"]["objective"]
    return {
        "experience_id": f"exp-{task['id']}-g{generation}-{new_best['agent']}",
        "source_task": task["id"],
        "family": task["family"],
        "task_signature": task["task_signature"],
        "failure_pattern": f"best candidate plateaued at generation {generation - 1} with objective {previous_objective}",
        "successful_strategy": new_best["strategy"],
        "tool_trace_summary": f"{new_best['agent']} improved {task['objective_label']} from {previous_objective} to {new_objective}",
        "delta_J": delta_j,
        "reusable_rules": new_best["reusable_rules"],
        "supporting_memory_ids": new_best["supporting_memory_ids"],
        "code_pattern": new_best["label"],
    }


def run_discrete_task(
    task: dict[str, Any],
    store: MemoryStore,
    *,
    epsilon: float = 0.05,
    progress_callback: ProgressCallback | None = None,
    pace_ms: int = 0,
) -> dict[str, Any]:
    _emit(progress_callback, pace_ms, phase="task_loaded", task_id=task["id"], message=f"Loaded {task['id']}")

    baseline = _baseline_candidate(task)
    baseline_objective = baseline["metrics"]["objective"]
    archive = [baseline]
    current_best = baseline

    generations = []
    objective_curve = [{"generation": 0, "objective": baseline_objective, "J": baseline["metrics"]["J"], "agent": "baseline"}]
    memory_events = []
    initial_retrieved = store.retrieve(task_signature=task["task_signature"], family=task["family"], top_k=4)

    for generation in range(1, int(task["instance"]["generations"]) + 1):
        generation_memories = store.retrieve(task_signature=task["task_signature"], family=task["family"], top_k=4)
        _emit(
            progress_callback,
            pace_ms,
            phase="generation_started",
            task_id=task["id"],
            generation=generation,
            message=f"Generation {generation} retrieved {len(generation_memories)} memories",
        )

        specs = tsp_specs(generation, generation_memories) if task["task_kind"] == "tsp" else max_cut_specs(generation, generation_memories)
        candidates = []
        for spec in specs:
            candidate = solve_candidate(task, spec, current_best, archive, baseline_objective)
            candidate["candidate_id"] = f"{task['id']}-g{generation}-{spec['operator']}"
            candidate["visual"] = solution_visual(task, candidate["solution"])
            candidate["solution_summary"] = candidate_summary(task, candidate)
            candidates.append(candidate)
            _emit(
                progress_callback,
                pace_ms,
                phase="candidate_finished",
                task_id=task["id"],
                generation=generation,
                candidate=candidate["agent"],
                architecture=candidate["operator_family"],
                message=f"{candidate['agent']} objective={candidate['metrics']['objective']} J={candidate['metrics']['J']}",
            )

        candidates.sort(key=lambda item: item["metrics"]["J"], reverse=True)
        generation_winner = candidates[0]
        previous_best = current_best

        if is_better(task["objective_direction"], generation_winner["metrics"]["objective"], current_best["metrics"]["objective"]):
            current_best = generation_winner
            archive = _update_archive(task["objective_direction"], archive, generation_winner)

        delta_j = round(generation_winner["metrics"]["J"] - previous_best["metrics"]["J"], 4)
        wrote_memory = False
        new_experience = None

        if (
            generation_winner["metrics"]["feasible"]
            and is_better(task["objective_direction"], generation_winner["metrics"]["objective"], previous_best["metrics"]["objective"])
            and delta_j > epsilon
        ):
            new_experience = _build_experience(task, generation, previous_best, generation_winner, delta_j)
            wrote_memory = store.append(new_experience)
            if wrote_memory:
                memory_events.append(
                    {
                        "generation": generation,
                        "experience_id": new_experience["experience_id"],
                        "delta_J": delta_j,
                        "strategy": new_experience["successful_strategy"],
                    }
                )
                _emit(
                    progress_callback,
                    pace_ms,
                    phase="memory_writeback",
                    task_id=task["id"],
                    generation=generation,
                    candidate=generation_winner["agent"],
                    message=f"Write back {new_experience['experience_id']}",
                )

        objective_curve.append(
            {
                "generation": generation,
                "objective": current_best["metrics"]["objective"],
                "J": current_best["metrics"]["J"],
                "agent": current_best["agent"],
            }
        )
        generations.append(
            {
                "generation": generation,
                "retrieved_memories": generation_memories,
                "candidates": candidates,
                "winner": generation_winner,
                "best_after_generation": current_best,
                "delta_J": delta_j,
                "wrote_memory": wrote_memory,
                "new_experience": new_experience,
            }
        )
        _emit(
            progress_callback,
            pace_ms,
            phase="generation_finished",
            task_id=task["id"],
            generation=generation,
            candidate=current_best["agent"],
            message=f"Best objective after generation {generation}: {current_best['metrics']['objective']}",
        )

    run_delta_j = round(current_best["metrics"]["J"] - baseline["metrics"]["J"], 4)
    return {
        "task": {
            "id": task["id"],
            "title": task["title"],
            "description": task["description"],
            "family": task["family"],
            "task_kind": task["task_kind"],
            "objective_label": task["objective_label"],
            "objective_direction": task["objective_direction"],
            "source_type": task["source_type"],
            "operator_menu": task["operator_menu"],
        },
        "baseline": baseline,
        "initial_retrieved_memories": initial_retrieved,
        "generations": generations,
        "winner": current_best,
        "delta_J": run_delta_j,
        "objective_curve": objective_curve,
        "memory_events": memory_events,
        "selection_reason": (
            f"{current_best['agent']} reached {task['objective_label']}={current_best['metrics']['objective']} "
            f"with J={current_best['metrics']['J']} after {len(generations)} generations."
        ),
    }
