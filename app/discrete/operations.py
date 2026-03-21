from __future__ import annotations

import math
import time
from typing import Any

from app.discrete.evals import score_candidate


def distance(left: dict[str, float], right: dict[str, float]) -> float:
    return math.hypot(left["x"] - right["x"], left["y"] - right["y"])


def tsp_length(route: list[int], nodes: list[dict[str, float]]) -> float:
    total = 0.0
    for index, node_id in enumerate(route):
        left = nodes[node_id]
        right = nodes[route[(index + 1) % len(route)]]
        total += distance(left, right)
    return total


def tsp_valid(route: list[int], node_count: int) -> bool:
    return sorted(route) == list(range(node_count))


def nearest_neighbor_route(nodes: list[dict[str, float]], start: int = 0) -> list[int]:
    unvisited = {node["id"] for node in nodes}
    route = [start]
    unvisited.remove(start)
    while unvisited:
        current = route[-1]
        next_node = min(unvisited, key=lambda node_id: distance(nodes[current], nodes[node_id]))
        route.append(next_node)
        unvisited.remove(next_node)
    return route


def two_opt(route: list[int], nodes: list[dict[str, float]], max_passes: int = 4) -> list[int]:
    best = route[:]
    best_length = tsp_length(best, nodes)
    for _ in range(max_passes):
        improved = False
        for left in range(1, len(best) - 2):
            for right in range(left + 1, len(best) - 1):
                candidate = best[:left] + list(reversed(best[left:right + 1])) + best[right + 1 :]
                candidate_length = tsp_length(candidate, nodes)
                if candidate_length + 1e-9 < best_length:
                    best = candidate
                    best_length = candidate_length
                    improved = True
        if not improved:
            break
    return best


def _segment_crosses(
    left_a: dict[str, float],
    left_b: dict[str, float],
    right_a: dict[str, float],
    right_b: dict[str, float],
) -> bool:
    def ccw(p1: dict[str, float], p2: dict[str, float], p3: dict[str, float]) -> float:
        return (p3["y"] - p1["y"]) * (p2["x"] - p1["x"]) - (p2["y"] - p1["y"]) * (p3["x"] - p1["x"])

    d1 = ccw(left_a, right_a, right_b)
    d2 = ccw(left_b, right_a, right_b)
    d3 = ccw(left_a, left_b, right_a)
    d4 = ccw(left_a, left_b, right_b)
    return d1 * d2 < 0 and d3 * d4 < 0


def crossing_repair(route: list[int], nodes: list[dict[str, float]]) -> list[int]:
    candidate = route[:]
    size = len(candidate)
    for left in range(size - 3):
        a = candidate[left]
        b = candidate[left + 1]
        for right in range(left + 2, size - 1):
            c = candidate[right]
            d = candidate[right + 1]
            if len({a, b, c, d}) < 4:
                continue
            if _segment_crosses(nodes[a], nodes[b], nodes[c], nodes[d]):
                return candidate[: left + 1] + list(reversed(candidate[left + 1 : right + 1])) + candidate[right + 1 :]
    return two_opt(candidate, nodes, max_passes=1)


def double_bridge(route: list[int]) -> list[int]:
    size = len(route)
    if size < 8:
        return route[:]
    cuts = [size // 4, size // 2, (3 * size) // 4]
    a = route[: cuts[0]]
    b = route[cuts[0] : cuts[1]]
    c = route[cuts[1] : cuts[2]]
    d = route[cuts[2] :]
    return a + c + b + d


def _edge_pairs(route: list[int]) -> set[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    for index, node_id in enumerate(route):
        edge = tuple(sorted((node_id, route[(index + 1) % len(route)])))
        pairs.add(edge)
    return pairs


def archive_edge_crossover(
    parent: list[int],
    archive: list[dict[str, Any]],
    nodes: list[dict[str, float]],
) -> list[int]:
    if len(archive) < 2:
        return two_opt(parent, nodes, max_passes=2)

    preferred_edges = _edge_pairs(archive[0]["solution"]) | _edge_pairs(archive[1]["solution"])
    unvisited = {node["id"] for node in nodes}
    route = [parent[0]]
    unvisited.remove(parent[0])
    while unvisited:
        current = route[-1]
        preferred = [
            node_id
            for node_id in unvisited
            if tuple(sorted((current, node_id))) in preferred_edges
        ]
        pool = preferred or list(unvisited)
        next_node = min(pool, key=lambda node_id: distance(nodes[current], nodes[node_id]))
        route.append(next_node)
        unvisited.remove(next_node)
    return two_opt(route, nodes, max_passes=2)


def cut_weight(assignment: list[int], edges: list[tuple[int, int, int]]) -> float:
    return float(sum(weight for left, right, weight in edges if assignment[left] != assignment[right]))


def max_cut_valid(assignment: list[int], node_count: int) -> bool:
    return len(assignment) == node_count and all(value in {0, 1} for value in assignment)


def _node_gain(assignment: list[int], node_id: int, edges: list[tuple[int, int, int]]) -> float:
    before = cut_weight(assignment, edges)
    flipped = assignment[:]
    flipped[node_id] = 1 - flipped[node_id]
    after = cut_weight(flipped, edges)
    return after - before


def heavy_edge_seed(node_count: int, edges: list[tuple[int, int, int]]) -> list[int]:
    assignment: list[int | None] = [None] * node_count
    for left, right, _weight in sorted(edges, key=lambda item: item[2], reverse=True):
        if assignment[left] is None and assignment[right] is None:
            assignment[left] = 0
            assignment[right] = 1
        elif assignment[left] is None:
            assignment[left] = 1 - assignment[right]
        elif assignment[right] is None:
            assignment[right] = 1 - assignment[left]
    for index, value in enumerate(assignment):
        if value is None:
            assignment[index] = index % 2
    return [int(value) for value in assignment]


def greedy_gain_flips(assignment: list[int], edges: list[tuple[int, int, int]]) -> list[int]:
    candidate = assignment[:]
    while True:
        gains = [(_node_gain(candidate, node_id, edges), node_id) for node_id in range(len(candidate))]
        best_gain, best_node = max(gains)
        if best_gain <= 1e-9:
            return candidate
        candidate[best_node] = 1 - candidate[best_node]


def pair_flip_escape(assignment: list[int], edges: list[tuple[int, int, int]]) -> list[int]:
    best = assignment[:]
    best_weight = cut_weight(best, edges)
    for left in range(len(best) - 1):
        for right in range(left + 1, len(best)):
            candidate = best[:]
            candidate[left] = 1 - candidate[left]
            candidate[right] = 1 - candidate[right]
            weight = cut_weight(candidate, edges)
            if weight > best_weight:
                best = candidate
                best_weight = weight
    return greedy_gain_flips(best, edges)


def community_swap(assignment: list[int], edges: list[tuple[int, int, int]]) -> list[int]:
    scores: list[tuple[float, int]] = []
    for node_id in range(len(assignment)):
        penalty = 0.0
        for left, right, weight in edges:
            if left == node_id:
                penalty += weight if assignment[left] == assignment[right] else -weight
            if right == node_id:
                penalty += weight if assignment[left] == assignment[right] else -weight
        scores.append((penalty, node_id))
    candidate = assignment[:]
    _score, node_id = max(scores)
    candidate[node_id] = 1 - candidate[node_id]
    return greedy_gain_flips(candidate, edges)


def archive_relink(assignment: list[int], archive: list[dict[str, Any]], edges: list[tuple[int, int, int]]) -> list[int]:
    if len(archive) < 2:
        return greedy_gain_flips(assignment, edges)
    target = archive[1]["solution"]
    candidate = assignment[:]
    differing = [index for index, (left, right) in enumerate(zip(candidate, target)) if left != right]
    while differing:
        best_gain = 0.0
        best_node: int | None = None
        for node_id in differing:
            trial = candidate[:]
            trial[node_id] = target[node_id]
            gain = cut_weight(trial, edges) - cut_weight(candidate, edges)
            if gain > best_gain:
                best_gain = gain
                best_node = node_id
        if best_node is None:
            break
        candidate[best_node] = target[best_node]
        differing = [index for index in differing if index != best_node]
    return greedy_gain_flips(candidate, edges)


def memory_support(spec: dict[str, Any], memories: list[dict[str, Any]]) -> list[str]:
    if not spec.get("uses_memory"):
        return []
    required_rules = set(spec.get("required_rules", []))
    if not required_rules:
        return [item["experience_id"] for item in memories]
    support_ids = []
    for item in memories:
        if required_rules & set(item.get("reusable_rules", [])):
            support_ids.append(item["experience_id"])
    return support_ids


def tsp_specs(generation: int, memories: list[dict[str, Any]]) -> list[dict[str, Any]]:
    specs = [
        {
            "operator": "nearest-neighbor-seed",
            "label": "Nearest-neighbor constructive seed",
            "family": "constructive",
            "strategy": "Build a route greedily from spatial proximity before any repair.",
            "complexity": 0.24,
            "uses_memory": False,
            "required_rules": [],
            "reusable_rules": ["construct_then_intensify"],
        },
        {
            "operator": "two-opt-local-search",
            "label": "2-opt local search",
            "family": "local-search",
            "strategy": "Reverse route segments whenever the swap shortens the tour.",
            "complexity": 0.28,
            "uses_memory": False,
            "required_rules": [],
            "reusable_rules": ["two_opt_after_seed"],
        },
        {
            "operator": "crossing-repair",
            "label": "Memory-guided crossing repair",
            "family": "memory-guided",
            "strategy": "Use replayed route-repair rules to remove obvious geometric crossings early.",
            "complexity": 0.33,
            "uses_memory": True,
            "required_rules": ["repair_crossings"],
            "reusable_rules": ["repair_crossings", "construct_then_intensify"],
        },
    ]
    if generation >= 2:
        specs.extend(
            [
                {
                    "operator": "double-bridge-kick",
                    "label": "Plateau-escape bridge kick",
                    "family": "exploration",
                    "strategy": "Apply a non-local perturbation, then intensify from the perturbed route.",
                    "complexity": 0.41,
                    "uses_memory": True,
                    "required_rules": ["escape_plateau"],
                    "reusable_rules": ["escape_plateau", "perturb_then_intensify"],
                },
                {
                    "operator": "archive-edge-crossover",
                    "label": "Archive edge crossover",
                    "family": "recombination",
                    "strategy": "Bias construction toward edges that survived in the archive, then repair locally.",
                    "complexity": 0.38,
                    "uses_memory": False,
                    "required_rules": [],
                    "reusable_rules": ["archive_recombination"],
                },
            ]
        )
    available = []
    for spec in specs:
        support_ids = memory_support(spec, memories)
        if spec["uses_memory"] and not support_ids:
            continue
        available.append({**spec, "supporting_memory_ids": support_ids})
    return available


def max_cut_specs(generation: int, memories: list[dict[str, Any]]) -> list[dict[str, Any]]:
    specs = [
        {
            "operator": "heavy-edge-seed",
            "label": "Heavy-edge constructive seed",
            "family": "constructive",
            "strategy": "Seed the partition by separating the heaviest edges first.",
            "complexity": 0.23,
            "uses_memory": True,
            "required_rules": ["heavy_edge_seed"],
            "reusable_rules": ["heavy_edge_seed"],
        },
        {
            "operator": "greedy-gain-flips",
            "label": "Greedy node-flip descent",
            "family": "local-search",
            "strategy": "Repeatedly flip the node with the largest positive gain in cut weight.",
            "complexity": 0.25,
            "uses_memory": True,
            "required_rules": ["greedy_flip_gain"],
            "reusable_rules": ["greedy_flip_gain"],
        },
        {
            "operator": "community-swap",
            "label": "Community pressure swap",
            "family": "memory-guided",
            "strategy": "Flip the node with the largest same-side penalty, then re-run greedy refinement.",
            "complexity": 0.34,
            "uses_memory": False,
            "required_rules": [],
            "reusable_rules": ["community_swap"],
        },
    ]
    if generation >= 2:
        specs.extend(
            [
                {
                    "operator": "pair-flip-escape",
                    "label": "Pair-flip plateau escape",
                    "family": "exploration",
                    "strategy": "Escape local minima with the best pair move before another greedy pass.",
                    "complexity": 0.39,
                    "uses_memory": True,
                    "required_rules": ["pair_flip_escape", "escape_plateau"],
                    "reusable_rules": ["pair_flip_escape", "escape_plateau"],
                },
                {
                    "operator": "archive-relink",
                    "label": "Archive relinking",
                    "family": "recombination",
                    "strategy": "Relink the current partition toward an archived alternative whenever that improves the cut.",
                    "complexity": 0.36,
                    "uses_memory": False,
                    "required_rules": [],
                    "reusable_rules": ["archive_recombination"],
                },
            ]
        )
    available = []
    for spec in specs:
        support_ids = memory_support(spec, memories)
        if spec["uses_memory"] and not support_ids:
            continue
        available.append({**spec, "supporting_memory_ids": support_ids})
    return available


def solve_candidate(
    task: dict[str, Any],
    spec: dict[str, Any],
    current_best: dict[str, Any],
    archive: list[dict[str, Any]],
    baseline_objective: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    if task["task_kind"] == "tsp":
        nodes = task["instance"]["nodes"]
        parent = current_best["solution"]
        if spec["operator"] == "nearest-neighbor-seed":
            solution = nearest_neighbor_route(nodes)
        elif spec["operator"] == "two-opt-local-search":
            solution = two_opt(parent, nodes, max_passes=4)
        elif spec["operator"] == "crossing-repair":
            solution = crossing_repair(parent, nodes)
            solution = two_opt(solution, nodes, max_passes=2)
        elif spec["operator"] == "double-bridge-kick":
            solution = two_opt(double_bridge(parent), nodes, max_passes=3)
        elif spec["operator"] == "archive-edge-crossover":
            solution = archive_edge_crossover(parent, archive, nodes)
        else:  # pragma: no cover
            raise ValueError(f"Unknown TSP operator: {spec['operator']}")
        feasible = tsp_valid(solution, len(nodes))
        objective = tsp_length(solution, nodes) if feasible else float("inf")
    elif task["task_kind"] == "max-cut":
        edges = task["instance"]["edges"]
        node_count = len(task["instance"]["nodes"])
        parent = current_best["solution"]
        if spec["operator"] == "heavy-edge-seed":
            solution = heavy_edge_seed(node_count, edges)
        elif spec["operator"] == "greedy-gain-flips":
            solution = greedy_gain_flips(parent, edges)
        elif spec["operator"] == "pair-flip-escape":
            solution = pair_flip_escape(parent, edges)
        elif spec["operator"] == "community-swap":
            solution = community_swap(parent, edges)
        elif spec["operator"] == "archive-relink":
            solution = archive_relink(parent, archive, edges)
        else:  # pragma: no cover
            raise ValueError(f"Unknown Max-Cut operator: {spec['operator']}")
        feasible = max_cut_valid(solution, node_count)
        objective = cut_weight(solution, edges) if feasible else 0.0
    else:  # pragma: no cover
        raise ValueError(f"Unknown task kind: {task['task_kind']}")

    latency_ms = (time.perf_counter() - started) * 1000.0
    metrics = score_candidate(
        direction=task["objective_direction"],
        objective=objective,
        baseline_objective=baseline_objective,
        feasible=feasible,
        latency_ms=latency_ms,
        complexity=spec["complexity"],
        memory_applied=bool(spec["supporting_memory_ids"]),
    )
    return {
        "agent": spec["operator"],
        "label": spec["label"],
        "operator_family": spec["family"],
        "strategy": spec["strategy"],
        "solution": solution,
        "supporting_memory_ids": spec["supporting_memory_ids"],
        "reusable_rules": spec["reusable_rules"],
        "metrics": metrics,
        "parent_ids": [archive[0]["candidate_id"]] if archive else [current_best["candidate_id"]],
    }
