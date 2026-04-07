from __future__ import annotations

from pathlib import Path

from app.bench.personalization_support import evaluate_label_candidate


def evaluate_candidate(*, task, candidate_path, source_code, baseline_metrics, memory_applied):
    del source_code, baseline_metrics, memory_applied
    return evaluate_label_candidate(task=task, candidate_path=Path(candidate_path))
