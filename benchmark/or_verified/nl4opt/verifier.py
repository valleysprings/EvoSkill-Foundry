from __future__ import annotations

from pathlib import Path

from app.bench.or_benchmarks import evaluate_coptpy_value_candidate, run_coptpy_value_benchmark


def evaluate_candidate(
    *,
    task,
    candidate_path,
    source_code,
    baseline_metrics,
    memory_applied,
):
    workspace_root = Path(candidate_path).parent / "_external"
    workspace_root.mkdir(parents=True, exist_ok=True)
    return evaluate_coptpy_value_candidate(
        task=task,
        candidate_path=candidate_path,
        source_code=source_code,
        workspace_root=workspace_root,
        max_items=task.get("runtime_max_items"),
        dataset_name="CardinalOperations/NL4OPT",
        dataset_split="test",
        summary_label="NL4OPT",
    )


def run_benchmark_adapter_task(
    *,
    task,
    candidate_path,
    source_code,
    proposal_runtime,
    workspace_root,
    session_id,
    max_items,
    progress_callback,
    pace_ms,
):
    return run_coptpy_value_benchmark(
        task=task,
        candidate_path=candidate_path,
        source_code=source_code,
        proposal_runtime=proposal_runtime,
        workspace_root=workspace_root,
        max_items=max_items,
        dataset_name="CardinalOperations/NL4OPT",
        dataset_split="test",
        summary_label="NL4OPT",
        progress_callback=progress_callback,
        pace_ms=pace_ms,
    )
