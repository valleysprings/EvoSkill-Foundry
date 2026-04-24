from __future__ import annotations

import json
from pathlib import Path

from app.bench.livecodebench_official_support import evaluate_livecodebench_problem
from app.bench.runtime_support import effective_suite_run_config


ROOT = Path(__file__).resolve().parent


def _problem_file(task: dict[str, object]) -> Path:
    item = dict(task.get("question_item") or {})
    metadata = dict(item.get("metadata") or {})
    relative_path = str(metadata.get("problem_file") or "").strip()
    if not relative_path:
        raise ValueError("LiveCodeBench question metadata is missing problem_file.")
    candidate_path = Path(relative_path)
    path = candidate_path if candidate_path.is_absolute() else ROOT / "data" / relative_path
    if not path.exists():
        raise FileNotFoundError(f"LiveCodeBench cached problem file was not found: {path}")
    return path


def _load_problem(task: dict[str, object]) -> dict[str, object]:
    return json.loads(_problem_file(task).read_text())


def evaluate_candidate(*, task, candidate_path, source_code, baseline_metrics, memory_applied):
    del source_code, baseline_metrics, memory_applied
    problem = _load_problem(task)
    if not list(problem.get("public_test_cases") or []) and not list(problem.get("private_test_cases") or []):
        raise ValueError("LiveCodeBench cached item did not contain any test cases.")
    config = effective_suite_run_config(task, Path(candidate_path))
    max_test_cases_raw = config.get("max_test_cases")
    max_test_cases = int(max_test_cases_raw) if isinstance(max_test_cases_raw, (int, float)) and max_test_cases_raw > 0 else None
    return evaluate_livecodebench_problem(problem, Path(candidate_path), max_test_cases=max_test_cases)
