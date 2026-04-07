from __future__ import annotations

from pathlib import Path

from app.bench.benchmark_adapter_support import (
    build_benchmark_adapter_candidate,
    build_benchmark_adapter_result,
    effective_suite_run_config,
    resolve_benchmark_adapter_memory_root,
)
from app.bench.safety_support import evaluate_single_turn_harmful_suite


TASK_ROOT = Path(__file__).resolve().parent
BENCHMARK_NAME = "harmbench"


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
    return evaluate_single_turn_harmful_suite(
        task=task,
        task_root=TASK_ROOT,
        candidate_path=Path(candidate_path),
        suite_config=config,
        benchmark_name=BENCHMARK_NAME,
        memory_root=resolve_benchmark_adapter_memory_root(task),
    )


def run_benchmark_adapter_task(
    *,
    task,
    candidate_path,
    source_code,
    proposal_runtime,
    workspace_root,
    memory_root,
    session_id,
    max_items,
    max_episodes,
    progress_callback,
    pace_ms,
):
    del workspace_root, session_id, max_episodes
    config = effective_suite_run_config(task, Path(candidate_path))
    raw_metrics = evaluate_single_turn_harmful_suite(
        task=task,
        task_root=TASK_ROOT,
        candidate_path=Path(candidate_path),
        proposal_runtime=proposal_runtime,
        suite_config=config,
        benchmark_name=BENCHMARK_NAME,
        max_items=max_items,
        memory_root=memory_root,
        progress_callback=progress_callback,
        pace_ms=pace_ms,
    )
    baseline = build_benchmark_adapter_candidate(
        task=task,
        source_code=source_code,
        agent="checked-in-wrapper",
        label="checked-in-wrapper",
        strategy="Use the checked-in pass-through wrapper with no extra defense logic.",
        rationale="The checked-in file provides the raw target-model baseline for HarmBench.",
        candidate_summary="Checked-in HarmBench pass-through wrapper.",
        raw_metrics={"status": "not-run", "verifier_status": "not-run"},
        workspace_path=str(candidate_path),
    )
    winner = build_benchmark_adapter_candidate(
        task=task,
        source_code=source_code,
        agent="target-model-defense",
        label="target-model-defense",
        strategy="Evaluate the candidate wrapper against fixed HarmBench jailbreak prompts.",
        rationale="A behavior counts as jailbroken if any fixed prompt yields harmful assistance.",
        candidate_summary="HarmBench harmful single-turn evaluation run.",
        raw_metrics=raw_metrics,
        workspace_path=str(candidate_path),
        proposal_model=proposal_runtime.active_model,
    )
    total = int(raw_metrics.get("total_tests") or 0)
    asr = float(raw_metrics.get("objective") or 0.0)
    return build_benchmark_adapter_result(
        task=task,
        proposal_runtime=proposal_runtime,
        baseline=baseline,
        winner=winner,
        selection_reason=f"Evaluated {total} HarmBench harmful behaviors with attack_success_rate={asr:.6f}.",
        extra_fields={
            "suite_summary": dict(raw_metrics.get("suite_summary") or {}),
            "item_runs": list(raw_metrics.get("item_runs") or []),
            "llm_traces": list(raw_metrics.get("llm_traces") or []),
            "memory_before_count": int(raw_metrics.get("memory_before_count") or 0),
            "memory_after_count": int(raw_metrics.get("memory_after_count") or 0),
        },
    )
