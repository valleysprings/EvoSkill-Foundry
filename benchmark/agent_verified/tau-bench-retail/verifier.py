from __future__ import annotations

from pathlib import Path

from app.codegen.agent_benchmarks import evaluate_tau_bench_candidate, run_tau_bench_suite


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
    return evaluate_tau_bench_candidate(
        task=task,
        candidate_path=candidate_path,
        source_code=source_code,
        workspace_root=workspace_root,
        session_id=Path(candidate_path).parent.name,
        max_items=task.get("runtime_max_items"),
        env_name="retail",
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
    return run_tau_bench_suite(
        task=task,
        candidate_path=candidate_path,
        source_code=source_code,
        proposal_runtime=proposal_runtime,
        workspace_root=workspace_root,
        session_id=session_id,
        max_items=max_items,
        env_name="retail",
        progress_callback=progress_callback,
        pace_ms=pace_ms,
    )
