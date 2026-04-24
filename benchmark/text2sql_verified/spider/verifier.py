from __future__ import annotations

from benchmark.text2sql_verified.text2sql_support import evaluate_spider_execution_candidate


def evaluate_candidate(*, task, candidate_path, source_code, baseline_metrics, memory_applied):
    return evaluate_spider_execution_candidate(task=task, candidate_path=candidate_path)
