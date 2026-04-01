from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from co_bench_support import evaluate_co_bench_candidate


def evaluate_candidate(
    *,
    task,
    candidate_path,
    source_code,
    baseline_metrics,
    memory_applied,
):
    return evaluate_co_bench_candidate(
        task=task,
        candidate_path=candidate_path,
        source_code=source_code,
    )
