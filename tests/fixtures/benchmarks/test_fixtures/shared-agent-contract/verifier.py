from __future__ import annotations

from pathlib import Path

from app.bench.agent_benchmarks import run_scripted_multi_turn_suite
from app.bench.benchmark_adapter_support import (
    build_benchmark_adapter_candidate,
    build_benchmark_adapter_result,
    effective_suite_run_config,
    runtime_for_benchmark_adapter_task,
)


def evaluate_candidate(
    *,
    task,
    candidate_path,
    source_code,
    baseline_metrics,
    memory_applied,
):
    del source_code, baseline_metrics, memory_applied
    config = effective_suite_run_config(task, Path(candidate_path))
    runtime = runtime_for_benchmark_adapter_task(task)
    return run_scripted_multi_turn_suite(
        task=task,
        candidate_path=Path(candidate_path),
        proposal_runtime=runtime,
        suite_name="fixture-suite",
        domain="fixture",
        scripted_episodes=list(config.get("inline_episodes") or []),
        suite_config=config,
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
    del workspace_root, session_id, max_items, progress_callback, pace_ms
    raw_metrics = evaluate_candidate(
        task=task,
        candidate_path=candidate_path,
        source_code=source_code,
        baseline_metrics=None,
        memory_applied=False,
    )
    if isinstance(max_episodes, int) and max_episodes > 0:
        raw_metrics["item_runs"] = list(raw_metrics.get("item_runs") or [])[:max_episodes]
        raw_metrics["test_results"] = list(raw_metrics.get("test_results") or [])[:max_episodes]
        raw_metrics["passed_tests"] = sum(1 for item in raw_metrics["item_runs"] if item.get("success"))
        raw_metrics["total_tests"] = len(raw_metrics["item_runs"])
        raw_metrics["objective"] = raw_metrics["passed_tests"] / raw_metrics["total_tests"] if raw_metrics["total_tests"] else 0.0
        raw_metrics["objective_score"] = raw_metrics["objective"]
        raw_metrics["objective_signal"] = raw_metrics["objective"]
    baseline = build_benchmark_adapter_candidate(
        task=task,
        source_code=source_code,
        agent="checked-in-adapter",
        label="checked-in-adapter",
        strategy="Use the checked-in shared-agent fixture adapter.",
        rationale="The checked-in file implements the multi-turn adapter baseline.",
        candidate_summary="Checked-in multi-turn fixture adapter.",
        raw_metrics={"status": "not-run", "verifier_status": "not-run"},
        workspace_path=str(candidate_path),
    )
    winner = build_benchmark_adapter_candidate(
        task=task,
        source_code=source_code,
        agent="fixture-agent",
        label="fixture-agent",
        strategy="Run the candidate adapter against the scripted multi-turn fixture.",
        rationale="The fixture succeeds only when the adapter emits the expected completion tool call.",
        candidate_summary="Scripted shared-agent-contract fixture run.",
        raw_metrics=raw_metrics,
        workspace_path=str(candidate_path),
        proposal_model=proposal_runtime.active_model,
    )
    passed = int(raw_metrics.get("passed_tests") or 0)
    total = int(raw_metrics.get("total_tests") or 0)
    return build_benchmark_adapter_result(
        task=task,
        proposal_runtime=proposal_runtime,
        baseline=baseline,
        winner=winner,
        selection_reason=f"Shared-agent fixture finished with {passed}/{total} successful episodes.",
        extra_fields={
            "suite_summary": dict(raw_metrics.get("suite_summary") or {}),
            "item_runs": list(raw_metrics.get("item_runs") or []),
        },
    )
