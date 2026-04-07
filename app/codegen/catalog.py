from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.configs.codegen import (
    DEFAULT_BRANCHING_FACTOR,
    DEFAULT_EDITABLE_FILE,
    DEFAULT_ENTRY_SYMBOL,
    REQUIRED_TASK_FIELDS,
    SEED_STRATEGY_EXPERIENCES,
    VALID_BENCHMARK_TIERS,
    speedup_objective_spec,
)
from app.bench.benchmark_adapter_support import uses_benchmark_adapter_runtime, load_value_from_candidate
from app.codegen.selection import selection_spec_for_task
from app.codegen.task_contracts import (
    infer_interaction_mode,
    infer_optimization_scope,
    infer_runtime_backend,
    infer_scoring_mode,
    infer_task_mode,
    infer_task_shape,
)
from app.configs.paths import BENCHMARK_ROOT as CONFIG_BENCHMARK_ROOT
from app.configs.paths import REGISTRY_PATH as CONFIG_REGISTRY_PATH

BENCHMARK_ROOT = CONFIG_BENCHMARK_ROOT
REGISTRY_PATH = CONFIG_REGISTRY_PATH

TRACK_ORDER = {
    "math_verified": 0,
    "reasoning_verified": 1,
    "text2sql_verified": 2,
    "longcontext_verified": 3,
    "personalization_verified": 4,
    "safety_verified": 5,
    "browse_snapshot": 6,
    "science_verified": 7,
    "coding_verified": 8,
    "or_verified": 9,
    "agent_verified": 10,
}
VALID_SAFETY_CATEGORIES = frozenset(
    {
        "jailbreak_attack",
        "over_refusal",
        "factuality_hallucination",
        "policy_drift",
        "benign_utility",
    }
)
VALID_SAFETY_FOCUS = frozenset(set(VALID_SAFETY_CATEGORIES) | {"should_refuse", "safety_degradation"})
TASK_ORDER = {
    "olymmath": 0,
    "math-500": 1,
    "aime-2024": 2,
    "aime-2025": 3,
    "aime-2026": 4,
    "planbench": 5,
    "arc-challenge": 6,
    "bbh": 7,
    "mmlu-pro": 8,
    "longbench-v2": 9,
    "incharacter": 11,
    "characterbench": 12,
    "socialbench": 13,
    "timechara": 14,
    "rmtbench": 16,
    "personamem-32k": 17,
    "personafeedback": 18,
    "alpsbench": 19,
    "alpbench": 20,
    "xstest-refusal-calibration": 21,
    "harmbench-text-harmful": 22,
    "jailbreakbench-harmful": 23,
    "or-bench-hard-1k": 24,
    "or-bench-toxic": 25,
    "hallulens-precisewikiqa": 26,
    "hallulens-mixedentities": 27,
    "hallulens-longwiki": 28,
    "longsafety": 29,
    "tom-gibbs-multiturn-jailbreak": 30,
    "safemtdata-benign-utility": 31,
    "tau-bench-retail": 32,
    "tau-bench-airline": 33,
    "sciq": 34,
    "qasc": 35,
    "scienceqa": 36,
    "openbookqa": 37,
    "gpqa-diamond": 38,
    "livecodebench": 39,
    "co-bench": 40,
    "alfworld": 41,
    "assistantbench": 42,
    "gaia": 43,
    "gaia2": 44,
    "osworld": 45,
}

def _speedup_objective_spec() -> dict[str, str]:
    return speedup_objective_spec()


def _count_manifest_items(path: Path) -> int:
    payload = json.loads(path.read_text())
    rows = payload.get("items") if isinstance(payload, dict) and "items" in payload else payload
    if not isinstance(rows, list):
        raise ValueError(f"Question manifest must contain a list of items: {path}")
    return len(rows)


def _infer_safety_category(task: dict[str, Any]) -> str | None:
    if str(task.get("track") or "").strip() != "safety_verified":
        return None
    task_id = str(task.get("id") or "").strip().lower()
    inferred: dict[str, str] = {
        "xstest-refusal-calibration": "over_refusal",
        "harmbench-text-harmful": "jailbreak_attack",
        "jailbreakbench-harmful": "jailbreak_attack",
        "or-bench-hard-1k": "over_refusal",
        "or-bench-toxic": "jailbreak_attack",
        "hallulens-precisewikiqa": "factuality_hallucination",
        "hallulens-mixedentities": "factuality_hallucination",
        "hallulens-longwiki": "factuality_hallucination",
        "longsafety": "jailbreak_attack",
        "tom-gibbs-multiturn-jailbreak": "policy_drift",
        "safemtdata-benign-utility": "benign_utility",
        "tau-bench-retail": "policy_drift",
        "tau-bench-airline": "policy_drift",
    }
    return inferred.get(task_id, "safety_degradation")


def _infer_safety_focus(task: dict[str, Any]) -> str | None:
    if str(task.get("track") or "").strip() != "safety_verified":
        return None
    task_id = str(task.get("id") or "").strip().lower()
    if task_id == "or-bench-toxic":
        return "should_refuse"
    if task_id == "longsafety":
        return "safety_degradation"
    return _infer_safety_category(task)


def _normalize_task(task: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(task)
    missing = [field for field in REQUIRED_TASK_FIELDS if not isinstance(normalized.get(field), str) or not str(normalized.get(field)).strip()]
    if missing:
        raise ValueError(f"Task {normalized.get('id') or '<unknown>'} is missing required fields: {', '.join(missing)}")

    benchmark_tier = str(normalized["benchmark_tier"]).strip()
    if benchmark_tier not in VALID_BENCHMARK_TIERS:
        raise ValueError(
            f"Task {normalized.get('id') or '<unknown>'} has invalid benchmark_tier={benchmark_tier!r}; "
            f"expected one of {sorted(VALID_BENCHMARK_TIERS)}."
        )

    objective_spec = dict(task.get("objective_spec") or {})
    if not objective_spec:
        objective_spec = _speedup_objective_spec()
    normalized["objective_spec"] = objective_spec
    normalized["objective_label"] = normalized.get("objective_label") or objective_spec["display_name"]
    normalized["objective_direction"] = normalized.get("objective_direction") or objective_spec["direction"]
    normalized["branching_factor"] = int(normalized.get("branching_factor", DEFAULT_BRANCHING_FACTOR))
    normalized["benchmark_tier"] = benchmark_tier
    normalized["track"] = str(normalized["track"]).strip()
    normalized["answer_metric"] = str(normalized["answer_metric"]).strip()
    normalized["dataset_id"] = str(normalized.get("dataset_id") or normalized["id"])
    normalized["dataset_size"] = int(normalized.get("dataset_size") or 0)
    normalized["entry_symbol"] = str(normalized.get("entry_symbol") or normalized.get("function_name") or DEFAULT_ENTRY_SYMBOL)
    normalized["function_name"] = str(normalized.get("function_name") or normalized["entry_symbol"])
    normalized["editable_file"] = str(normalized.get("editable_file") or DEFAULT_EDITABLE_FILE)
    normalized["editable_filename"] = Path(normalized["editable_file"]).name
    included_in_main_comparison = normalized.get("included_in_main_comparison")
    if included_in_main_comparison is None:
        normalized["included_in_main_comparison"] = True
    elif isinstance(included_in_main_comparison, bool):
        normalized["included_in_main_comparison"] = included_in_main_comparison
    else:
        raise ValueError(
            f"Task {normalized.get('id') or '<unknown>'} has invalid included_in_main_comparison="
            f"{included_in_main_comparison!r}; expected a boolean when provided."
        )
    normalized["runtime_backend"] = infer_runtime_backend(normalized)
    normalized["task_mode"] = infer_task_mode(normalized)
    normalized["interaction_mode"] = infer_interaction_mode(normalized)
    normalized["optimization_scope"] = infer_optimization_scope(normalized)
    normalized["task_shape"] = infer_task_shape(normalized)
    normalized["scoring_mode"] = infer_scoring_mode(normalized)
    research_line = str(normalized.get("research_line") or "").strip()
    if not research_line:
        research_line = "personalization" if normalized["track"] == "personalization_verified" else "general"
    normalized["research_line"] = research_line
    personalization_category = str(normalized.get("personalization_category") or "").strip()
    normalized["personalization_category"] = personalization_category or None
    personalization_focus = str(normalized.get("personalization_focus") or "").strip()
    normalized["personalization_focus"] = personalization_focus or None
    safety_category = str(normalized.get("safety_category") or "").strip()
    if not safety_category:
        safety_category = str(_infer_safety_category(normalized) or "").strip()
    normalized["safety_category"] = safety_category or None
    safety_focus = str(normalized.get("safety_focus") or "").strip()
    if not safety_focus:
        safety_focus = str(_infer_safety_focus(normalized) or "").strip()
    normalized["safety_focus"] = safety_focus or None
    if normalized["track"] == "safety_verified":
        if normalized["safety_category"] not in VALID_SAFETY_CATEGORIES:
            raise ValueError(
                f"Task {normalized['id']} must declare safety_category in {sorted(VALID_SAFETY_CATEGORIES)}."
            )
        if normalized["safety_focus"] not in VALID_SAFETY_FOCUS:
            raise ValueError(
                f"Task {normalized['id']} must declare safety_focus in {sorted(VALID_SAFETY_FOCUS)}."
            )
    else:
        if normalized["safety_category"] is not None and normalized["safety_category"] not in VALID_SAFETY_CATEGORIES:
            raise ValueError(
                f"Task {normalized['id']} has invalid safety_category={normalized['safety_category']!r}; "
                f"expected one of {sorted(VALID_SAFETY_CATEGORIES)}."
            )
        if normalized["safety_focus"] is not None and normalized["safety_focus"] not in VALID_SAFETY_FOCUS:
            raise ValueError(
                f"Task {normalized['id']} has invalid safety_focus={normalized['safety_focus']!r}; "
                f"expected one of {sorted(VALID_SAFETY_FOCUS)}."
            )
    supports_eval_model = normalized.get("supports_eval_model")
    if supports_eval_model is None:
        normalized["supports_eval_model"] = False
    elif isinstance(supports_eval_model, bool):
        normalized["supports_eval_model"] = supports_eval_model
    else:
        raise ValueError(
            f"Task {normalized.get('id') or '<unknown>'} has invalid supports_eval_model={supports_eval_model!r}; "
            "expected a boolean when provided."
        )
    requires_eval_model = normalized.get("requires_eval_model")
    if requires_eval_model is None:
        normalized["requires_eval_model"] = False
    elif isinstance(requires_eval_model, bool):
        normalized["requires_eval_model"] = requires_eval_model
    else:
        raise ValueError(
            f"Task {normalized.get('id') or '<unknown>'} has invalid requires_eval_model={requires_eval_model!r}; "
            "expected a boolean when provided."
        )
    if normalized["requires_eval_model"] and not normalized["supports_eval_model"]:
        raise ValueError(
            f"Task {normalized['id']} declares requires_eval_model=true but supports_eval_model is not true."
        )
    default_eval_model = normalized.get("default_eval_model")
    if default_eval_model is None:
        normalized["default_eval_model"] = None
    else:
        parsed_default_eval_model = str(default_eval_model).strip()
        normalized["default_eval_model"] = parsed_default_eval_model or None
    if normalized["default_eval_model"] and not normalized["supports_eval_model"]:
        raise ValueError(
            f"Task {normalized['id']} declares default_eval_model but does not support eval_model."
        )
    normalized["local_dataset_only"] = bool(normalized.get("local_dataset_only"))
    split = normalized.get("split")
    normalized["split"] = str(split).strip() if isinstance(split, str) and split.strip() else None
    item_manifest = normalized.get("item_manifest")
    normalized["item_manifest"] = str(item_manifest).strip() if isinstance(item_manifest, str) and item_manifest.strip() else None
    normalized["lazy_item_manifest"] = bool(normalized.get("lazy_item_manifest"))
    normalized["prompt_context"] = str(normalized.get("prompt_context") or "")
    normalized["allow_browsing"] = bool(normalized.get("allow_browsing", False))
    raw_run_baseline_verifier = normalized.get("run_baseline_verifier")
    if raw_run_baseline_verifier is None:
        normalized["run_baseline_verifier"] = normalized["runtime_backend"] != "benchmark_adapter"
    else:
        normalized["run_baseline_verifier"] = bool(raw_run_baseline_verifier)
    normalized["verifier_path"] = str(normalized["verifier_path"])
    normalized["editable_path"] = str(normalized["editable_path"])
    normalized["selection_spec"] = selection_spec_for_task(normalized)
    if normalized["runtime_backend"] != "dataset" and normalized["local_dataset_only"]:
        raise ValueError(
            f"Task {normalized['id']} declares local_dataset_only=true but runtime_backend={normalized['runtime_backend']!r}."
        )
    if normalized["local_dataset_only"]:
        if normalized["dataset_size"] <= 0:
            raise ValueError(f"Dataset task {normalized['id']} must declare dataset_size > 0.")
        if normalized["item_manifest"] is None:
            raise ValueError(f"Dataset task {normalized['id']} must declare item_manifest.")
    return normalized


def _suite_run_config(task: dict[str, Any]) -> dict[str, Any] | None:
    if not uses_benchmark_adapter_runtime(task):
        return None
    config = dict(load_value_from_candidate(Path(str(task["editable_path"])), "RUN_CONFIG", {}) or {})
    build_run_config = load_value_from_candidate(Path(str(task["editable_path"])), "build_run_config", None)
    if callable(build_run_config):
        built = build_run_config()
        if isinstance(built, dict):
            config = dict(built)
    override = task.get("runtime_suite_config")
    if isinstance(override, dict):
        config.update(override)
    return config


def _suite_default_max_items(config: dict[str, Any] | None) -> int | None:
    if not config:
        return None
    for key in ("task_limit", "n_tasks", "cases"):
        value = config.get(key)
        if isinstance(value, bool):
            continue
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            return parsed
    for key in ("task_ids", "problem_names", "task_names", "tasks", "inline_episodes"):
        value = config.get(key)
        if isinstance(value, list) and value:
            return len(value)
    return None


def _suite_default_max_episodes(config: dict[str, Any] | None) -> int | None:
    if not config:
        return None
    for key in ("episode_limit", "n_episodes", "max_episodes", "task_limit"):
        value = config.get(key)
        if isinstance(value, bool):
            continue
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            return parsed
    for key in ("episode_ids", "episodes", "inline_episodes"):
        value = config.get(key)
        if isinstance(value, list) and value:
            return len(value)
    return None


def _task_supports_max_items(task: dict[str, Any]) -> bool:
    if bool(task.get("local_dataset_only")):
        return True
    return uses_benchmark_adapter_runtime(task) and task.get("interaction_mode") == "single_turn"


def _task_default_max_items(task: dict[str, Any], suite_run_config: dict[str, Any] | None) -> int | None:
    if bool(task.get("local_dataset_only")):
        size = int(task.get("dataset_size") or 0)
        return size if size > 0 else None
    if uses_benchmark_adapter_runtime(task) and task.get("interaction_mode") == "single_turn":
        return _suite_default_max_items(suite_run_config)
    return None


def _task_supports_max_episodes(task: dict[str, Any]) -> bool:
    return uses_benchmark_adapter_runtime(task) and task.get("interaction_mode") == "multi_turn"


def _task_default_max_episodes(task: dict[str, Any], suite_run_config: dict[str, Any] | None) -> int | None:
    if not _task_supports_max_episodes(task):
        return None
    return _suite_default_max_episodes(suite_run_config)


def task_summary(task: dict[str, Any]) -> dict[str, Any]:
    suite_run_config = _suite_run_config(task)
    default_item_workers = 0 if uses_benchmark_adapter_runtime(task) else 20
    return {
        "id": task["id"],
        "title": task["title"],
        "description": task["description"],
        "family": task["family"],
        "function_name": task["function_name"],
        "entry_symbol": task["entry_symbol"],
        "editable_file": task["editable_file"],
        "answer_metric": task["answer_metric"],
        "objective_label": task["objective_label"],
        "objective_direction": task["objective_direction"],
        "objective_spec": task["objective_spec"],
        "selection_spec": task["selection_spec"],
        "generation_budget": task["generation_budget"],
        "candidate_budget": task["candidate_budget"],
        "branching_factor": task["branching_factor"],
        "item_workers": int(task.get("item_workers") or default_item_workers),
        "benchmark_tier": task["benchmark_tier"],
        "track": task["track"],
        "dataset_id": task["dataset_id"],
        "dataset_size": task["dataset_size"],
        "local_dataset_only": task["local_dataset_only"],
        "split": task["split"],
        "runtime_backend": task["runtime_backend"],
        "task_mode": task["task_mode"],
        "interaction_mode": task["interaction_mode"],
        "optimization_scope": task["optimization_scope"],
        "task_shape": task["task_shape"],
        "scoring_mode": task["scoring_mode"],
        "research_line": task["research_line"],
        "personalization_category": task["personalization_category"],
        "personalization_focus": task["personalization_focus"],
        "safety_category": task["safety_category"],
        "safety_focus": task["safety_focus"],
        "supports_eval_model": task["supports_eval_model"],
        "requires_eval_model": task["requires_eval_model"],
        "default_eval_model": task["default_eval_model"],
        "included_in_main_comparison": task["included_in_main_comparison"],
        "run_baseline_verifier": task["run_baseline_verifier"],
        "supports_runtime_config": suite_run_config is not None,
        "suite_run_config": suite_run_config,
        "supports_max_items": _task_supports_max_items(task),
        "default_max_items": _task_default_max_items(task, suite_run_config),
        "supports_max_episodes": _task_supports_max_episodes(task),
        "default_max_episodes": _task_default_max_episodes(task, suite_run_config),
    }


def _registry_entries() -> list[dict[str, Any]]:
    if not REGISTRY_PATH.exists():
        raise FileNotFoundError(f"Benchmark registry is missing: {REGISTRY_PATH}")
    payload = json.loads(REGISTRY_PATH.read_text())
    entries = payload.get("tasks")
    if not isinstance(entries, list):
        raise ValueError("benchmark/registry.json must contain a top-level 'tasks' list.")
    return [dict(entry) for entry in entries]


def _load_task(entry: dict[str, Any]) -> dict[str, Any]:
    relative_path = entry.get("path")
    if not isinstance(relative_path, str) or not relative_path.strip():
        raise ValueError("Every benchmark registry entry must declare a non-empty path.")
    task_dir = BENCHMARK_ROOT / relative_path
    task_path = task_dir / "task.json"
    if not task_path.exists():
        raise FileNotFoundError(f"Task spec not found: {task_path}")
    task = json.loads(task_path.read_text())
    if not isinstance(task, dict):
        raise ValueError(f"Task spec must be a JSON object: {task_path}")
    track_from_path = Path(relative_path).parts[0]
    declared_track = str(task.get("track") or "").strip()
    if declared_track and declared_track != track_from_path:
        raise ValueError(
            f"Task {task.get('id') or '<unknown>'} declares track={declared_track!r} "
            f"but registry path lives under {track_from_path!r}."
        )
    merged = {**task, "task_dir": str(task_dir), "task_path": str(task_path)}
    merged["editable_path"] = str(task_dir / str(task.get("editable_file") or ""))
    merged["verifier_path"] = str(task_dir / str(task.get("verifier") or ""))
    if not Path(merged["editable_path"]).exists():
        raise FileNotFoundError(f"Editable file not found: {merged['editable_path']}")
    if not Path(merged["verifier_path"]).exists():
        raise FileNotFoundError(f"Verifier file not found: {merged['verifier_path']}")
    item_manifest = task.get("item_manifest")
    if isinstance(item_manifest, str) and item_manifest.strip():
        item_manifest_path = task_dir / item_manifest
        if not item_manifest_path.exists() and not bool(task.get("lazy_item_manifest")):
            raise FileNotFoundError(f"Question manifest not found: {item_manifest_path}")
        merged["item_manifest_path"] = str(item_manifest_path)
        if item_manifest_path.exists():
            prepared_item_count = _count_manifest_items(item_manifest_path)
            merged["prepared_item_count"] = prepared_item_count
            if not bool(task.get("lazy_item_manifest")):
                merged["dataset_size"] = prepared_item_count
    data_file = task.get("data_file")
    if isinstance(data_file, str) and data_file.strip():
        merged["data_path"] = str(task_dir / data_file)
        merged["data"] = json.loads((task_dir / data_file).read_text())
    readme_path = task_dir / "README.md"
    if readme_path.exists():
        merged["readme_path"] = str(readme_path)
    return _normalize_task(merged)


def list_missing_local_dataset_warnings() -> list[dict[str, str]]:
    warnings: list[dict[str, str]] = []
    for entry in _registry_entries():
        if not bool(entry.get("enabled", True)):
            continue
        relative_path = str(entry.get("path") or "").strip()
        if not relative_path:
            continue
        task_dir = BENCHMARK_ROOT / relative_path
        task_path = task_dir / "task.json"
        if not task_path.exists():
            continue
        try:
            payload = json.loads(task_path.read_text())
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict) or not bool(payload.get("local_dataset_only")):
            continue
        item_manifest = str(payload.get("item_manifest") or "").strip()
        if not item_manifest:
            continue
        manifest_path = task_dir / item_manifest
        if manifest_path.exists():
            continue
        task_id = str(payload.get("id") or entry.get("id") or "").strip() or str(entry.get("id") or "")
        title = str(payload.get("title") or task_id).strip() or task_id
        track = str(payload.get("track") or Path(relative_path).parts[0]).strip() or "unknown"
        warnings.append(
            {
                "task_id": task_id,
                "title": title,
                "track": track,
                "manifest_path": str(manifest_path),
                "prepare_command": f"python benchmark/prepare_datasets.py --task-id {task_id}",
                "message": f"Missing local dataset manifest: {manifest_path}",
            }
        )
    return warnings


def _sort_key(task: dict[str, Any]) -> tuple[int, int, int, str, str]:
    return (
        0 if task["included_in_main_comparison"] else 1,
        TRACK_ORDER.get(task["track"], len(TRACK_ORDER)),
        TASK_ORDER.get(task["id"], 999),
        task["track"],
        task["id"],
    )

def load_codegen_tasks(
    task_id: str | None = None,
    *,
    included_in_main_comparison: bool | None = None,
) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for entry in _registry_entries():
        if not bool(entry.get("enabled", True)):
            continue
        try:
            tasks.append(_load_task(entry))
        except FileNotFoundError:
            continue
    if task_id is not None:
        tasks = [task for task in tasks if task["id"] == task_id]
    if included_in_main_comparison is not None:
        tasks = [task for task in tasks if task["included_in_main_comparison"] == included_in_main_comparison]
    return sorted(tasks, key=_sort_key)


def list_codegen_task_summaries() -> list[dict[str, Any]]:
    return [task_summary(task) for task in load_codegen_tasks()]


def seed_strategy_experiences() -> list[dict[str, Any]]:
    return [dict(item) for item in SEED_STRATEGY_EXPERIENCES]
