from __future__ import annotations

from pathlib import Path

from app.bench.safety_official_support import evaluate_longsafety_candidate


def evaluate_candidate(*, task, candidate_path, source_code, baseline_metrics, memory_applied):
    del source_code, baseline_metrics, memory_applied
    item = task.get("question_item")
    if not isinstance(item, dict):
        raise ValueError("Dataset question task must provide question_item.")
    return evaluate_longsafety_candidate(
        task=task,
        candidate_path=Path(candidate_path),
        item=item,
    )
