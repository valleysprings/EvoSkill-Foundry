from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


BENCHMARK_ROOT = Path(__file__).resolve().parent
DATA_SOURCES_DOC = BENCHMARK_ROOT / "README.md"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run benchmark-local dataset prepare.py scripts.")
    parser.add_argument("--benchmark-root", default=str(BENCHMARK_ROOT))
    parser.add_argument("--registry")
    parser.add_argument("--task-id", action="append", dest="task_ids", default=[])
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--list", action="store_true", help="List benchmark tasks and whether they expose prepare.py.")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print per-task readiness details, including which tasks are ready or not ready.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the prepare commands without running them.")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep preparing later tasks even if one prepare.py fails.",
    )
    return parser.parse_args(argv)


def _load_registry_entries(registry_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(registry_path.read_text())
    entries = payload.get("tasks")
    if not isinstance(entries, list):
        raise ValueError(f"Registry must contain a top-level tasks list: {registry_path}")
    return [dict(entry) for entry in entries]


def _select_entries(entries: list[dict[str, Any]], task_ids: list[str]) -> list[dict[str, Any]]:
    by_id = {str(entry.get("id")): entry for entry in entries}
    if task_ids:
        missing = [task_id for task_id in task_ids if task_id not in by_id]
        if missing:
            raise ValueError(f"Unknown task ids: {', '.join(missing)}")
        return [by_id[task_id] for task_id in task_ids]
    return [entry for entry in entries if bool(entry.get("enabled", True))]


def _prepare_script_path(benchmark_root: Path, entry: dict[str, Any]) -> Path:
    relative_path = str(entry.get("path") or "").strip()
    if not relative_path:
        raise ValueError(f"Registry entry is missing path: {entry}")
    return benchmark_root / relative_path / "prepare.py"


def _task_dir(benchmark_root: Path, entry: dict[str, Any]) -> Path:
    relative_path = str(entry.get("path") or "").strip()
    if not relative_path:
        raise ValueError(f"Registry entry is missing path: {entry}")
    return benchmark_root / relative_path


def _task_spec_path(benchmark_root: Path, entry: dict[str, Any]) -> Path:
    return _task_dir(benchmark_root, entry) / "task.json"


def _maybe_load_task_spec(benchmark_root: Path, entry: dict[str, Any]) -> dict[str, Any] | None:
    task_path = _task_spec_path(benchmark_root, entry)
    if not task_path.exists():
        return None
    payload = json.loads(task_path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Task spec must contain an object: {task_path}")
    return payload


def _local_dataset_manifest_path(benchmark_root: Path, entry: dict[str, Any], task_spec: dict[str, Any] | None) -> Path | None:
    if not isinstance(task_spec, dict) or not bool(task_spec.get("local_dataset_only")):
        return None
    item_manifest = str(task_spec.get("item_manifest") or "").strip()
    if not item_manifest:
        return None
    return _task_dir(benchmark_root, entry) / item_manifest


def _declared_prepare_paths(benchmark_root: Path, entry: dict[str, Any], task_spec: dict[str, Any] | None) -> list[Path]:
    if not isinstance(task_spec, dict):
        return []
    raw_paths = task_spec.get("prepare_ready_paths")
    if not isinstance(raw_paths, list):
        return []
    task_dir = _task_dir(benchmark_root, entry)
    resolved: list[Path] = []
    for item in raw_paths:
        value = str(item or "").strip()
        if not value:
            continue
        resolved.append(task_dir / value)
    return resolved


def _dataset_status(benchmark_root: Path, entry: dict[str, Any]) -> tuple[str, Path | None]:
    task_spec = _maybe_load_task_spec(benchmark_root, entry)
    manifest_path = _local_dataset_manifest_path(benchmark_root, entry, task_spec)
    if manifest_path is not None:
        return ("yes" if manifest_path.exists() else "no"), manifest_path
    prepare_paths = _declared_prepare_paths(benchmark_root, entry, task_spec)
    if prepare_paths:
        missing = [path for path in prepare_paths if not path.exists()]
        return ("yes" if not missing else "no"), (missing[0] if missing else prepare_paths[0])
    return "n/a", None


def _prepare_status(benchmark_root: Path, entry: dict[str, Any]) -> str:
    return "yes" if _prepare_script_path(benchmark_root, entry).exists() else "no"


def _debug_status_parts(benchmark_root: Path, entry: dict[str, Any]) -> tuple[str, list[str]]:
    task_id = str(entry["id"])
    prepare_status = _prepare_status(benchmark_root, entry)
    dataset_status, manifest_path = _dataset_status(benchmark_root, entry)
    path = str(entry["path"])

    status = "ready"
    note = "no local dataset preparation required"
    path_label = "path"
    path_value = path
    if dataset_status == "yes":
        note = "declared local dataset assets are present"
        if manifest_path is not None:
            path_label = "ready_path"
            path_value = str(manifest_path)
    elif dataset_status == "no":
        status = "not-ready"
        note = "declared local dataset assets are missing"
        if manifest_path is not None:
            path_label = "missing_path"
            path_value = str(manifest_path)
    elif prepare_status == "yes":
        status = "unknown"
        note = "prepare.py exists but the task does not declare a readiness path"

    parts = [
        task_id,
        f"status={status}",
        f"prepare={prepare_status}",
        f"local_dataset_ready={dataset_status}",
        f"path={path}",
        f"{path_label}={path_value}",
        f"note={note}",
    ]
    if status == "not-ready":
        parts.append(f"prepare_command=python benchmark/prepare_datasets.py --task-id {task_id}")
    return status, parts


def _run_prepare_script(script_path: Path, python_executable: str, *, dry_run: bool) -> None:
    command = [python_executable, str(script_path)]
    print(f"[prepare] {' '.join(command)}")
    if dry_run:
        return
    completed = subprocess.run(
        command,
        cwd=script_path.parent,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.stdout.strip():
        print(completed.stdout.strip())
    if completed.stderr.strip():
        print(completed.stderr.strip(), file=sys.stderr)
    if completed.returncode != 0:
        raise RuntimeError(f"{script_path} exited with code {completed.returncode}")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    benchmark_root = Path(args.benchmark_root).resolve()
    registry_path = Path(args.registry).resolve() if args.registry else benchmark_root / "registry.json"
    entries = _load_registry_entries(registry_path)
    selected = _select_entries(entries, list(args.task_ids))

    if args.debug:
        rows: list[tuple[str, list[str]]] = [_debug_status_parts(benchmark_root, entry) for entry in selected]
        ready_count = sum(1 for status, _ in rows if status == "ready")
        not_ready_count = sum(1 for status, _ in rows if status == "not-ready")
        unknown_count = sum(1 for status, _ in rows if status == "unknown")
        print(f"ready={ready_count} not_ready={not_ready_count} unknown={unknown_count}")
        for _, parts in rows:
            print("\t".join(parts))
        return 0

    if args.list:
        for entry in selected:
            prepare_status = _prepare_status(benchmark_root, entry)
            dataset_status, manifest_path = _dataset_status(benchmark_root, entry)
            status_parts = [
                str(entry["id"]),
                f"prepare={prepare_status}",
                f"local_dataset_ready={dataset_status}",
                f"path={entry['path']}",
            ]
            if manifest_path is not None:
                status_parts.append(f"manifest={manifest_path}")
            print("\t".join(status_parts))
        return 0

    prepared = 0
    skipped = 0
    failures: list[str] = []
    for entry in selected:
        script_path = _prepare_script_path(benchmark_root, entry)
        if not script_path.exists():
            dataset_status, manifest_path = _dataset_status(benchmark_root, entry)
            if dataset_status == "no":
                detail = (
                    f"enabled local dataset manifest is missing at {manifest_path} and no prepare.py was found"
                    if manifest_path is not None
                    else "enabled local dataset manifest is missing and no prepare.py was found"
                )
                detail = f"{detail}; see {DATA_SOURCES_DOC} for source locations"
                failures.append(f"{entry['id']}: {detail}")
                print(f"[error] {entry['id']}: {detail}", file=sys.stderr)
                if not args.continue_on_error:
                    break
                continue
            skipped += 1
            print(f"[skip] {entry['id']}: no prepare.py")
            continue
        try:
            _run_prepare_script(script_path, args.python, dry_run=bool(args.dry_run))
            prepared += 1
        except Exception as exc:  # noqa: BLE001
            detail = f"{exc}; see {DATA_SOURCES_DOC} for source locations"
            failures.append(f"{entry['id']}: {detail}")
            print(f"[error] {entry['id']}: {detail}", file=sys.stderr)
            if not args.continue_on_error:
                break

    print(f"prepared={prepared} skipped={skipped} failed={len(failures)}")
    if failures:
        print(f"See {DATA_SOURCES_DOC} for benchmark source datasets and manual download links.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
