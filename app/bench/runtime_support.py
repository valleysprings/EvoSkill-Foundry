from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from app.codegen.llm import ProposalRuntime
from app.configs.codegen import ITEM_MEMORY_JSON_NAME, ITEM_MEMORY_MD_NAME
from app.memory.store import MemoryStore


def _load_module_from_path(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Unable to import module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_candidate_module(path: Path):
    module_name = f"candidate_module_{path.parent.name}_{path.stem}".replace("-", "_")
    return _load_module_from_path(path, module_name)


def load_value_from_candidate(path: Path, name: str, default: Any = None) -> Any:
    module = load_candidate_module(path)
    return getattr(module, name, default)


def effective_suite_run_config(task: dict[str, Any], candidate_path: Path) -> dict[str, Any]:
    module = load_candidate_module(candidate_path)
    builder = getattr(module, "build_run_config", None)
    if callable(builder):
        base_config = builder()
    else:
        base_config = getattr(module, "RUN_CONFIG", {})
    if base_config is None:
        base = {}
    elif isinstance(base_config, dict):
        base = dict(base_config)
    else:
        raise ValueError("build_run_config()/RUN_CONFIG must produce a dict.")
    override = task.get("runtime_suite_config")
    if isinstance(override, dict):
        return {**base, **override}
    return base


def runtime_for_task(task: dict[str, Any]) -> ProposalRuntime:
    requested_model = str(task.get("runtime_model_override") or "").strip() or None
    return ProposalRuntime.from_env().with_model(requested_model)


def resolve_item_memory_root(
    task: dict[str, Any],
    *,
    memory_root: Path | str | None = None,
) -> Path | None:
    raw_value = memory_root if memory_root is not None else task.get("memory_root")
    if raw_value is None:
        return None
    if isinstance(raw_value, Path):
        return raw_value
    text = str(raw_value).strip()
    return Path(text) if text else None


def item_memory_store(
    task: dict[str, Any],
    *,
    item_id: str,
    memory_root: Path | str | None = None,
) -> MemoryStore | None:
    root = resolve_item_memory_root(task, memory_root=memory_root)
    if root is None:
        return None
    item_dir = root / str(task["id"]) / str(item_id)
    return MemoryStore(
        item_dir / ITEM_MEMORY_JSON_NAME,
        markdown_path=item_dir / ITEM_MEMORY_MD_NAME,
        title=f"{task['id']}:{item_id} Strategy Memory",
    )


def emit_progress(
    progress_callback,
    *,
    task_id: str,
    phase: str,
    message: str,
    **extra: Any,
) -> None:
    if progress_callback is None:
        return
    progress_callback(
        {
            "phase": phase,
            "task_id": task_id,
            "message": message,
            **extra,
        }
    )
