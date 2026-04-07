from __future__ import annotations

from pathlib import Path

from app.bench.agent_verified_support import (
    build_unavailable_result,
    load_benchmark_info,
    missing_runtime_metrics,
)


TASK_ROOT = Path(__file__).resolve().parent
BENCHMARK_NAME = "AssistantBench"


def _runtime_error() -> str:
    base = (
        "AssistantBench is registered as an agent_verified multi-turn task, but the official "
        "BrowserGym runtime bridge is not wired into this repo yet."
    )
    try:
        info = load_benchmark_info(TASK_ROOT)
    except FileNotFoundError:
        return f"{base} Run prepare.py first to create task-local data/benchmark_info.json."
    status = str(info.get("runtime_status") or "").strip()
    return f"{base} {status}".strip()


def evaluate_candidate(
    *,
    task,
    candidate_path,
    source_code,
    baseline_metrics,
    memory_applied,
):
    del task, candidate_path, source_code, baseline_metrics, memory_applied
    return missing_runtime_metrics(error=_runtime_error())


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
    del workspace_root, session_id, max_items, max_episodes, progress_callback, pace_ms
    return build_unavailable_result(
        task=task,
        candidate_path=Path(candidate_path),
        source_code=source_code,
        proposal_runtime=proposal_runtime,
        benchmark_name=BENCHMARK_NAME,
        error=_runtime_error(),
    )
