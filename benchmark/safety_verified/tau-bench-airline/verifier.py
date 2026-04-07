from __future__ import annotations

import tempfile
from pathlib import Path

from app.bench.agent_benchmarks import evaluate_tau_bench_candidate, run_tau_bench_suite


TASK_ROOT = Path(__file__).resolve().parent
ENV_NAME = "airline"


def evaluate_candidate(
    *,
    task,
    candidate_path,
    source_code,
    baseline_metrics,
    memory_applied,
):
    del source_code, baseline_metrics, memory_applied
    with tempfile.TemporaryDirectory() as tmp_dir:
        return evaluate_tau_bench_candidate(
            task=task,
            candidate_path=Path(candidate_path),
            source_code="",
            workspace_root=Path(tmp_dir),
            session_id="tau-bench-airline-eval",
            max_items=None,
            max_episodes=None,
            env_name=ENV_NAME,
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
    max_episodes,
    progress_callback,
    pace_ms,
):
    return run_tau_bench_suite(
        task=task,
        candidate_path=Path(candidate_path),
        source_code=source_code,
        proposal_runtime=proposal_runtime,
        workspace_root=Path(workspace_root),
        session_id=session_id,
        max_items=max_items,
        max_episodes=max_episodes,
        env_name=ENV_NAME,
        progress_callback=progress_callback,
        pace_ms=pace_ms,
    )
