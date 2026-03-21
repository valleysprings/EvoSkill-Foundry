from __future__ import annotations

from typing import Any


TSP_COORDS = [
    {"id": 0, "x": 82, "y": 94},
    {"id": 1, "x": 126, "y": 58},
    {"id": 2, "x": 168, "y": 88},
    {"id": 3, "x": 212, "y": 64},
    {"id": 4, "x": 252, "y": 112},
    {"id": 5, "x": 298, "y": 86},
    {"id": 6, "x": 324, "y": 138},
    {"id": 7, "x": 286, "y": 188},
    {"id": 8, "x": 232, "y": 204},
    {"id": 9, "x": 182, "y": 172},
    {"id": 10, "x": 138, "y": 212},
    {"id": 11, "x": 96, "y": 178},
    {"id": 12, "x": 74, "y": 236},
    {"id": 13, "x": 148, "y": 266},
    {"id": 14, "x": 218, "y": 258},
    {"id": 15, "x": 274, "y": 246},
    {"id": 16, "x": 332, "y": 222},
    {"id": 17, "x": 352, "y": 82},
]

MAXCUT_NODES = [
    {"id": 0, "x": 78, "y": 88},
    {"id": 1, "x": 132, "y": 58},
    {"id": 2, "x": 160, "y": 116},
    {"id": 3, "x": 104, "y": 164},
    {"id": 4, "x": 152, "y": 214},
    {"id": 5, "x": 82, "y": 244},
    {"id": 6, "x": 268, "y": 78},
    {"id": 7, "x": 324, "y": 56},
    {"id": 8, "x": 348, "y": 114},
    {"id": 9, "x": 292, "y": 164},
    {"id": 10, "x": 340, "y": 220},
    {"id": 11, "x": 270, "y": 246},
]

MAXCUT_EDGES = [
    (0, 1, 8), (0, 2, 6), (0, 3, 7), (1, 2, 5), (1, 3, 6), (2, 4, 7),
    (3, 4, 8), (3, 5, 6), (4, 5, 7), (0, 5, 5), (1, 4, 4),
    (6, 7, 8), (6, 8, 6), (6, 9, 7), (7, 8, 7), (7, 9, 5), (8, 10, 8),
    (9, 10, 6), (9, 11, 7), (10, 11, 8), (6, 11, 5), (8, 11, 4),
    (0, 6, 10), (1, 7, 12), (2, 8, 11), (3, 9, 13), (4, 10, 12), (5, 11, 11),
    (1, 8, 9), (2, 9, 10), (4, 9, 9), (3, 10, 8), (0, 7, 7), (5, 10, 7),
]


DISCRETE_TASKS: list[dict[str, Any]] = [
    {
        "id": "euclidean-tsp",
        "title": "Clustered Euclidean TSP",
        "description": "Optimize a 18-city Euclidean tour with constructive seeds, local repair, and plateau-escape operators.",
        "family": "route-search",
        "objective_label": "tour length",
        "objective_direction": "min",
        "task_signature": ["discrete-opt", "tsp", "euclidean", "clustered"],
        "task_kind": "tsp",
        "source_type": "synthetic-deterministic-instance",
        "instance": {
            "nodes": TSP_COORDS,
            "generations": 7,
        },
        "operator_menu": [
            "index-baseline",
            "nearest-neighbor-seed",
            "two-opt-local-search",
            "crossing-repair",
            "double-bridge-kick",
            "archive-edge-crossover",
        ],
    },
    {
        "id": "weighted-max-cut",
        "title": "Weighted Max-Cut with Community Structure",
        "description": "Optimize a weighted graph partition with greedy gain flips, pair escapes, and archive relinking.",
        "family": "graph-partition",
        "objective_label": "cut weight",
        "objective_direction": "max",
        "task_signature": ["discrete-opt", "max-cut", "weighted-graph", "community-structure"],
        "task_kind": "max-cut",
        "source_type": "synthetic-deterministic-instance",
        "instance": {
            "nodes": MAXCUT_NODES,
            "edges": MAXCUT_EDGES,
            "generations": 7,
        },
        "operator_menu": [
            "parity-baseline",
            "heavy-edge-seed",
            "greedy-gain-flips",
            "pair-flip-escape",
            "community-swap",
            "archive-relink",
        ],
    },
]


def load_discrete_tasks() -> list[dict[str, Any]]:
    return [dict(task) for task in DISCRETE_TASKS]


def list_discrete_task_summaries() -> list[dict[str, Any]]:
    return [
        {
            "id": task["id"],
            "title": task["title"],
            "description": task["description"],
            "family": task["family"],
            "objective_label": task["objective_label"],
            "objective_direction": task["objective_direction"],
            "task_kind": task["task_kind"],
            "source_type": task["source_type"],
            "operator_menu": task["operator_menu"],
        }
        for task in load_discrete_tasks()
    ]
