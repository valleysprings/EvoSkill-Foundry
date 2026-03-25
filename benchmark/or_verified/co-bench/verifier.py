from __future__ import annotations

from pathlib import Path

from app.codegen.co_benchmarks import evaluate_co_bench_candidate, run_co_bench_suite


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
    return evaluate_co_bench_candidate(
        task=task,
        candidate_path=candidate_path,
        source_code=source_code,
        workspace_root=workspace_root,
        max_items=task.get("runtime_max_items"),
    )


def run_external_task(
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
    return run_co_bench_suite(
        task=task,
        candidate_path=candidate_path,
        source_code=source_code,
        proposal_runtime=proposal_runtime,
        workspace_root=workspace_root,
        max_items=max_items,
        progress_callback=progress_callback,
        pace_ms=pace_ms,
    )
