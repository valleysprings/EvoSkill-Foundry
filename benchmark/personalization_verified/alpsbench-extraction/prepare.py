from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TRACK_ROOT = ROOT.parent
REPO_ROOT = ROOT.parents[2]
for candidate in (TRACK_ROOT, REPO_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from alpsbench_public import TASK1_DATASET_ID, TASK1_SIZE, add_items_argument, build_task1_items, requested_count, write_task_manifest


MANIFEST_PATH = ROOT / "data" / "questions.json"
SPLIT = "validation:task1"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize AlpsBench Task 1 validation items.")
    return add_items_argument(parser).parse_args(argv)


def _build_items(limit: int | None = None) -> list[dict]:
    return build_task1_items(benchmark="alpsbench-extraction", limit=limit)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    requested = requested_count(TASK1_SIZE, args.items)
    items = _build_items(limit=requested)
    write_task_manifest(
        manifest_path=MANIFEST_PATH,
        dataset_id=TASK1_DATASET_ID,
        split=SPLIT,
        items=items,
        dataset_size=TASK1_SIZE,
    )
    print(f"Wrote {len(items)} AlpsBench task1 rows to {MANIFEST_PATH}.")


if __name__ == "__main__":
    main()
