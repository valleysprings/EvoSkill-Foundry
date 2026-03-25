from __future__ import annotations

from app.codegen.external import (
    build_external_candidate,
    build_external_result,
    effective_external_run_config,
    render_external_run_config_source,
)


def evaluate_candidate(
    *,
    task,
    candidate_path,
    source_code,
    baseline_metrics,
    memory_applied,
):
    config = effective_external_run_config(task, candidate_path)
    total = int(config.get("cases", 0))
    passed = int(config.get("passed", 0))
    success_rate = passed / total if total else 0.0
    return {
        "status": "pass",
        "verifier_status": "pass",
        "correctness": success_rate,
        "passed_tests": passed,
        "total_tests": total,
        "objective": success_rate,
        "objective_score": success_rate,
        "objective_signal": success_rate,
        "test_results": [{"name": "fixture-summary", "actual": passed, "expected": total, "passed": passed == total}],
        "external_summary": {"passed": passed, "total": total},
    }


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
    config = effective_external_run_config(task, candidate_path)
    source_code = render_external_run_config_source(config)
    raw_metrics = evaluate_candidate(
        task=task,
        candidate_path=candidate_path,
        source_code=source_code,
        baseline_metrics=None,
        memory_applied=False,
    )
    total = int(raw_metrics.get("total_tests", 0))
    passed = int(raw_metrics.get("passed_tests", 0))

    baseline = build_external_candidate(
        task=task,
        source_code=source_code,
        agent="checked-in-config",
        label="checked-in-config",
        strategy="Use the checked-in external benchmark configuration.",
        rationale="This fixture establishes the no-op baseline for external benchmark runs.",
        candidate_summary="Checked-in external benchmark configuration.",
        raw_metrics={"status": "not-run", "verifier_status": "not-run"},
        workspace_path=str(candidate_path),
    )
    winner = build_external_candidate(
        task=task,
        source_code=source_code,
        agent=str(config.get("agent_name") or "fixture-agent"),
        label="fixture-agent",
        strategy="Run the external harness directly.",
        rationale="The benchmark returns a precomputed success ratio for regression testing.",
        candidate_summary="Fixture result from the external benchmark path.",
        raw_metrics=raw_metrics,
        workspace_path=str(workspace_root),
        proposal_model=proposal_runtime.active_model,
    )
    return build_external_result(
        task=task,
        proposal_runtime=proposal_runtime,
        baseline=baseline,
        winner=winner,
        selection_reason=f"External fixture finished with {passed}/{total} successful cases.",
        extra_fields={"external_summary": {"passed": passed, "total": total}},
    )
