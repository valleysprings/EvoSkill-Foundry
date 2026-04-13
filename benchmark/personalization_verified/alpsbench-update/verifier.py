from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TRACK_ROOT = ROOT.parent
REPO_ROOT = ROOT.parents[2]
for candidate in (TRACK_ROOT, REPO_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from alpsbench_public import evaluate_structured_candidate, score_task2, _task2_prediction_payload


def evaluate_candidate(*, task, candidate_path, source_code, baseline_metrics, memory_applied):
    del source_code, baseline_metrics, memory_applied
    return evaluate_structured_candidate(
        task=task,
        candidate_path=Path(candidate_path),
        parse_expectation='JSON memory update output: {"memory_items": [...]} or a JSON list',
        payload_builder=_task2_prediction_payload,
        score_fn=score_task2,
    )
