from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.configs.paths import BENCHMARK_ROOT
from app.codegen.task_contracts import SCORING_MODE_VALUES, TASK_SHAPE_VALUES


REFERENCE_CATALOG_PATH = BENCHMARK_ROOT / "personalization_verified" / "reference_benchmarks.json"
VALID_REFERENCE_STATUSES = {"local_task", "external_reference", "planned_task"}
VALID_REFERENCE_INTERACTION_MODES = {"single_turn", "multi_turn"}
VALID_REFERENCE_CATEGORIES = {
    "explicit_character_persona",
    "user_persona_personalization",
    "trait_behavior",
}
VALID_PRIMARY_CATEGORIES = {
    "character_knowledge",
    "character_portrayal",
    "consistency_robustness",
    "user_personalization",
    "agentic_personalization",
}
VALID_IMPLEMENTATION_STATUSES = {"running", "phase1", "phase2", "blocked"}
VALID_SUBJECT_DOMAINS = {
    "anime_acg",
    "games",
    "literary_fiction",
    "movie_tv",
    "general_fiction",
    "celebrity_real_person",
    "assistant_task_oriented",
}
VALID_METRIC_BACKENDS = {
    "deterministic_local",
    "llm_judge",
    "reward_model",
    "hybrid",
}
VALID_METRIC_FIDELITY = {
    "official",
    "adapted_local",
    "proxy_local",
    "reference_only",
}
VALID_RUNTIME_ROLES = {
    "policy_model",
    "judge_model",
    "evaluator_model",
    "reward_model",
    "interrogator_model",
    "env_model",
    "nsp_model",
}


def _required_string(entry: dict[str, Any], key: str) -> str:
    value = str(entry.get(key) or "").strip()
    if not value:
        raise ValueError(f"Personalization reference benchmark is missing required field {key!r}.")
    return value


def _optional_string(entry: dict[str, Any], key: str) -> str | None:
    value = str(entry.get(key) or "").strip()
    return value or None


def _optional_bool(entry: dict[str, Any], key: str) -> bool | None:
    raw_value = entry.get(key)
    if raw_value is None:
        return None
    if not isinstance(raw_value, bool):
        raise ValueError(f"Personalization reference benchmark field {key!r} must be a boolean when provided.")
    return raw_value


def _optional_string_list(entry: dict[str, Any], key: str) -> list[str]:
    raw_value = entry.get(key)
    if raw_value is None:
        return []
    if not isinstance(raw_value, list):
        raise ValueError(f"Personalization reference benchmark field {key!r} must be a list when provided.")
    values: list[str] = []
    seen: set[str] = set()
    for raw_item in raw_value:
        item = str(raw_item or "").strip()
        if not item:
            raise ValueError(f"Personalization reference benchmark field {key!r} cannot contain empty values.")
        if item in seen:
            continue
        values.append(item)
        seen.add(item)
    return values


def load_personalization_reference_benchmarks(path: Path | None = None) -> list[dict[str, Any]]:
    catalog_path = path or REFERENCE_CATALOG_PATH
    if not catalog_path.exists():
        raise FileNotFoundError(f"Missing personalization reference catalog: {catalog_path}")

    payload = json.loads(catalog_path.read_text())
    if not isinstance(payload, list):
        raise ValueError(f"Personalization reference catalog must contain a top-level list: {catalog_path}")

    references: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for raw_entry in payload:
        if not isinstance(raw_entry, dict):
            raise ValueError(f"Every personalization reference entry must be an object: {catalog_path}")

        entry = dict(raw_entry)
        benchmark_id = _required_string(entry, "id")
        if benchmark_id in seen_ids:
            raise ValueError(f"Duplicate personalization reference benchmark id={benchmark_id!r}.")
        seen_ids.add(benchmark_id)

        status = _required_string(entry, "status")
        if status not in VALID_REFERENCE_STATUSES:
            raise ValueError(
                f"Personalization reference benchmark {benchmark_id!r} has invalid status={status!r}; "
                f"expected one of {sorted(VALID_REFERENCE_STATUSES)}."
            )

        interaction_mode = _required_string(entry, "interaction_mode")
        if interaction_mode not in VALID_REFERENCE_INTERACTION_MODES:
            raise ValueError(
                f"Personalization reference benchmark {benchmark_id!r} has invalid interaction_mode={interaction_mode!r}; "
                f"expected one of {sorted(VALID_REFERENCE_INTERACTION_MODES)}."
            )

        benchmark_category = _required_string(entry, "benchmark_category")
        if benchmark_category not in VALID_REFERENCE_CATEGORIES:
            raise ValueError(
                f"Personalization reference benchmark {benchmark_id!r} has invalid benchmark_category={benchmark_category!r}; "
                f"expected one of {sorted(VALID_REFERENCE_CATEGORIES)}."
            )

        implementation_status = _required_string(entry, "implementation_status")
        if implementation_status not in VALID_IMPLEMENTATION_STATUSES:
            raise ValueError(
                f"Personalization reference benchmark {benchmark_id!r} has invalid implementation_status={implementation_status!r}; "
                f"expected one of {sorted(VALID_IMPLEMENTATION_STATUSES)}."
            )

        task_shape = _required_string(entry, "task_shape")
        if task_shape not in TASK_SHAPE_VALUES:
            raise ValueError(
                f"Personalization reference benchmark {benchmark_id!r} has invalid task_shape={task_shape!r}; "
                f"expected one of {sorted(TASK_SHAPE_VALUES)}."
            )

        scoring_mode = _required_string(entry, "scoring_mode")
        if scoring_mode not in SCORING_MODE_VALUES:
            raise ValueError(
                f"Personalization reference benchmark {benchmark_id!r} has invalid scoring_mode={scoring_mode!r}; "
                f"expected one of {sorted(SCORING_MODE_VALUES)}."
            )

        primary_category = _required_string(entry, "primary_category")
        if primary_category not in VALID_PRIMARY_CATEGORIES:
            raise ValueError(
                f"Personalization reference benchmark {benchmark_id!r} has invalid primary_category={primary_category!r}; "
                f"expected one of {sorted(VALID_PRIMARY_CATEGORIES)}."
            )

        subject_domains = _optional_string_list(entry, "subject_domains")
        invalid_subject_domains = [domain for domain in subject_domains if domain not in VALID_SUBJECT_DOMAINS]
        if invalid_subject_domains:
            raise ValueError(
                f"Personalization reference benchmark {benchmark_id!r} has invalid subject_domains={invalid_subject_domains!r}; "
                f"expected values drawn from {sorted(VALID_SUBJECT_DOMAINS)}."
            )

        official_metric_backend = _required_string(entry, "official_metric_backend")
        if official_metric_backend not in VALID_METRIC_BACKENDS:
            raise ValueError(
                f"Personalization reference benchmark {benchmark_id!r} has invalid official_metric_backend={official_metric_backend!r}; "
                f"expected one of {sorted(VALID_METRIC_BACKENDS)}."
            )

        metric_fidelity = _required_string(entry, "metric_fidelity")
        if metric_fidelity not in VALID_METRIC_FIDELITY:
            raise ValueError(
                f"Personalization reference benchmark {benchmark_id!r} has invalid metric_fidelity={metric_fidelity!r}; "
                f"expected one of {sorted(VALID_METRIC_FIDELITY)}."
            )

        official_dimensions = _optional_string_list(entry, "official_dimensions")
        if not official_dimensions:
            raise ValueError(
                f"Personalization reference benchmark {benchmark_id!r} must declare a non-empty official_dimensions list."
            )

        protocol_summary = _required_string(entry, "protocol_summary")
        implementation_note = _required_string(entry, "implementation_note")

        required_runtime_roles = _optional_string_list(entry, "required_runtime_roles")
        if not required_runtime_roles:
            raise ValueError(
                f"Personalization reference benchmark {benchmark_id!r} must declare a non-empty required_runtime_roles list."
            )
        invalid_runtime_roles = [role for role in required_runtime_roles if role not in VALID_RUNTIME_ROLES]
        if invalid_runtime_roles:
            raise ValueError(
                f"Personalization reference benchmark {benchmark_id!r} has invalid required_runtime_roles={invalid_runtime_roles!r}; "
                f"expected values drawn from {sorted(VALID_RUNTIME_ROLES)}."
            )

        supports_eval_model = _optional_bool(entry, "supports_eval_model")
        requires_eval_model = _optional_bool(entry, "requires_eval_model")
        default_eval_model = _optional_string(entry, "default_eval_model")
        if bool(requires_eval_model) and not bool(supports_eval_model):
            raise ValueError(
                f"Personalization reference benchmark {benchmark_id!r} requires_eval_model=true but supports_eval_model is not true."
            )
        if default_eval_model and not bool(supports_eval_model):
            raise ValueError(
                f"Personalization reference benchmark {benchmark_id!r} declares default_eval_model but does not support eval_model."
            )

        source = entry.get("source")
        if not isinstance(source, dict):
            raise ValueError(f"Personalization reference benchmark {benchmark_id!r} must declare a source object.")
        source_label = str(source.get("label") or "").strip()
        source_url = str(source.get("url") or "").strip()
        if not source_label or not source_url:
            raise ValueError(
                f"Personalization reference benchmark {benchmark_id!r} must declare non-empty source.label and source.url."
            )

        mirror = entry.get("mirror")
        mirror_slug: str | None = None
        mirror_url: str | None = None
        if mirror is not None:
            if not isinstance(mirror, dict):
                raise ValueError(f"Personalization reference benchmark {benchmark_id!r} mirror must be an object.")
            mirror_slug = str(mirror.get("slug") or "").strip() or None
            mirror_url = str(mirror.get("url") or "").strip() or None
            if bool(mirror_slug) ^ bool(mirror_url):
                raise ValueError(
                    f"Personalization reference benchmark {benchmark_id!r} mirror must provide both slug and url."
                )

        task_id = _optional_string(entry, "task_id")
        task_ids = _optional_string_list(entry, "task_ids")
        if task_id and task_ids and task_id not in task_ids:
            raise ValueError(
                f"Personalization reference benchmark {benchmark_id!r} declares task_id={task_id!r} "
                f"that is not present in task_ids."
            )
        normalized_task_ids: list[str] = []
        for candidate in [task_id, *task_ids]:
            if not candidate or candidate in normalized_task_ids:
                continue
            normalized_task_ids.append(candidate)
        blocking_reason = _optional_string(entry, "blocking_reason")
        if status == "local_task":
            if not normalized_task_ids:
                raise ValueError(
                    f"Local personalization reference benchmark {benchmark_id!r} must declare task_id or task_ids."
                )
            if implementation_status != "running":
                raise ValueError(
                    f"Local personalization reference benchmark {benchmark_id!r} must declare implementation_status='running'."
                )
            if blocking_reason:
                raise ValueError(
                    f"Local personalization reference benchmark {benchmark_id!r} cannot declare blocking_reason."
                )
            if metric_fidelity == "reference_only":
                raise ValueError(
                    f"Local personalization reference benchmark {benchmark_id!r} cannot declare metric_fidelity='reference_only'."
                )
        elif implementation_status == "running":
            raise ValueError(
                f"Only local_task personalization references may declare implementation_status='running'; got status={status!r} "
                f"for benchmark {benchmark_id!r}."
            )
        elif not blocking_reason and implementation_status == "blocked":
            raise ValueError(
                f"Blocked personalization reference benchmark {benchmark_id!r} must declare blocking_reason."
            )

        references.append(
            {
                "id": benchmark_id,
                "title": _required_string(entry, "title"),
                "status": status,
                "task_id": normalized_task_ids[0] if len(normalized_task_ids) == 1 else None,
                "task_ids": normalized_task_ids,
                "interaction_mode": interaction_mode,
                "benchmark_category": benchmark_category,
                "primary_category": primary_category,
                "secondary_categories": _optional_string_list(entry, "secondary_categories"),
                "subject_domains": subject_domains,
                "implementation_status": implementation_status,
                "task_shape": task_shape,
                "scoring_mode": scoring_mode,
                "supports_eval_model": bool(supports_eval_model),
                "requires_eval_model": bool(requires_eval_model),
                "default_eval_model": default_eval_model,
                "official_metric_name": _required_string(entry, "official_metric_name"),
                "official_metric_backend": official_metric_backend,
                "official_metric_granularity": _required_string(entry, "official_metric_granularity"),
                "metric_fidelity": metric_fidelity,
                "official_dimensions": official_dimensions,
                "protocol_summary": protocol_summary,
                "implementation_note": implementation_note,
                "required_runtime_roles": required_runtime_roles,
                "blocking_reason": blocking_reason,
                "focus": _required_string(entry, "focus"),
                "summary": _required_string(entry, "summary"),
                "source_label": source_label,
                "source_url": source_url,
                "mirror_slug": mirror_slug,
                "mirror_url": mirror_url,
            }
        )

    return references


def list_personalization_mirror_repos(path: Path | None = None) -> tuple[tuple[str, str], ...]:
    repos: list[tuple[str, str]] = []
    seen: set[str] = set()
    for entry in load_personalization_reference_benchmarks(path):
        slug = entry.get("mirror_slug")
        url = entry.get("mirror_url")
        if not slug or not url or slug in seen:
            continue
        repos.append((str(slug), str(url)))
        seen.add(str(slug))
    return tuple(repos)
