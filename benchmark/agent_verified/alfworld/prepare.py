from __future__ import annotations

import shutil
from pathlib import Path


TASK_ROOT = Path(__file__).resolve().parent
TASK_DATA_ROOT = TASK_ROOT / "data"
EXTERNAL_DATA_ROOT = TASK_ROOT.parents[2] / "external" / "alfworld" / "data"
REQUIRED_DIRS = ("json_2.1.1", "logic", "detectors")


def _materialize_directory(name: str) -> None:
    destination = TASK_DATA_ROOT / name
    source = EXTERNAL_DATA_ROOT / name

    if destination.is_symlink():
        destination.unlink()
    if destination.exists():
        return
    if not source.exists():
        raise FileNotFoundError(
            f"Missing ALFWorld asset {name!r}. Expected either {destination} or {source} to exist."
        )
    shutil.move(str(source), str(destination))


def main() -> int:
    TASK_DATA_ROOT.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED_DIRS:
        _materialize_directory(name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
