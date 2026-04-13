from __future__ import annotations

from pathlib import Path

from app.bench.safety_support import evaluate_single_turn_refusal_item_candidate


BENCHMARK_NAME = "hallulens-mixedentities"
DEFAULT_EXPECTED_BEHAVIOR = "refuse"


def evaluate_candidate(*, task, candidate_path, source_code, baseline_metrics, memory_applied):
    del source_code, baseline_metrics, memory_applied
    item = task.get("question_item")
    if not isinstance(item, dict):
        raise ValueError("Dataset question task must provide question_item.")
    return evaluate_single_turn_refusal_item_candidate(
        task=task,
        candidate_path=Path(candidate_path),
        item=item,
        benchmark_name=BENCHMARK_NAME,
        default_expected_behavior=DEFAULT_EXPECTED_BEHAVIOR,
    )
