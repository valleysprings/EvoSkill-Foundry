from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
TASK_ROOT = Path(__file__).resolve().parent
TASK_DATA_ROOT = TASK_ROOT / "data"
TASK_JSON_ROOT = TASK_DATA_ROOT / "json_2.1.1"
TASK_MANIFEST_PATH = TASK_DATA_ROOT / "questions.json"
TASK_SOURCE_INFO_PATH = TASK_DATA_ROOT / "source_info.json"
EXTERNAL_DATA_ROOT = ROOT / "external" / "alfworld" / "data"
DEFAULT_SPLIT = "valid_seen"
DEFAULT_TASK_LIMIT = 140
REQUIRED_DIRS = ("json_2.1.1", "logic", "detectors")


def _materialize_directory(name: str) -> None:
    destination = TASK_DATA_ROOT / name
    if destination.exists():
        return

    source = EXTERNAL_DATA_ROOT / name
    if not source.exists():
        raise FileNotFoundError(
            f"Missing ALFWorld asset {name!r}. Expected either {destination} or {source} to exist."
        )
    shutil.copytree(source, destination)


def _instruction_from_traj(traj_payload: dict[str, Any]) -> str:
    anns = list((traj_payload.get("turk_annotations") or {}).get("anns") or [])
    for ann in anns:
        if not isinstance(ann, dict):
            continue
        task_desc = str(ann.get("task_desc") or "").strip()
        if task_desc:
            return task_desc
    task_type = str(traj_payload.get("task_type") or "alfworld task").replace("_", " ").strip()
    return task_type.capitalize()


def _episode_specs(split: str = DEFAULT_SPLIT) -> list[dict[str, Any]]:
    split_root = TASK_JSON_ROOT / split
    if not split_root.exists():
        raise FileNotFoundError(f"ALFWorld split directory not found: {split_root}")

    rows: list[dict[str, Any]] = []
    for game_path in sorted(split_root.rglob("game.tw-pddl")):
        parent_text = str(game_path.parent)
        if "movable" in parent_text or "Sliced" in parent_text:
            continue

        traj_path = game_path.with_name("traj_data.json")
        if not traj_path.exists():
            continue

        game_payload = json.loads(game_path.read_text())
        if not bool(game_payload.get("solvable")):
            continue

        traj_payload = json.loads(traj_path.read_text())
        episode_id = "/".join(game_path.relative_to(split_root).parts[:-1])
        instruction = _instruction_from_traj(traj_payload)
        rows.append(
            {
                "episode_id": episode_id,
                "instruction": instruction,
                "game_file": str(game_path.relative_to(TASK_ROOT)),
                "traj_file": str(traj_path.relative_to(TASK_ROOT)),
                "split": split,
            }
        )

    if not rows:
        raise ValueError(f"No solvable ALFWorld episodes found under {split_root}")
    return rows


def _manifest_payload(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    for episode in episodes:
        episode_id = str(episode["episode_id"])
        instruction = str(episode["instruction"])
        items.append(
            {
                "item_id": episode_id,
                "name": episode_id,
                "prompt": instruction,
                "context": {
                    "suite": "alfworld-text",
                    "domain": "alfworld",
                    "episode_id": episode_id,
                    "split": str(episode["split"]),
                },
                "expected_answer": "Solve episode",
                "metadata": {
                    "benchmark": "alfworld",
                    "episode_id": episode_id,
                    "instruction": instruction,
                    "split": str(episode["split"]),
                    "game_file": str(episode["game_file"]),
                    "traj_file": str(episode["traj_file"]),
                },
            }
        )
    return {
        "dataset_id": "alfworld_text_valid_seen",
        "dataset_size": DEFAULT_TASK_LIMIT,
        "prepared_count": len(items),
        "items": items,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare the local ALFWorld dataset manifest.")
    parser.add_argument("--items", type=int, default=None, help="Optional minimum item count to materialize.")
    args = parser.parse_args(argv)
    del args

    TASK_DATA_ROOT.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED_DIRS:
        _materialize_directory(name)

    episodes = _episode_specs()[:DEFAULT_TASK_LIMIT]
    _write_json(TASK_MANIFEST_PATH, _manifest_payload(episodes))
    _write_json(
        TASK_SOURCE_INFO_PATH,
        {
            "dataset_id": "alfworld_text_valid_seen",
            "split": DEFAULT_SPLIT,
            "prepared_count": len(episodes),
            "source": "task_local_alfworld_assets",
            "task_root": str(TASK_ROOT),
            "json_root": str(TASK_JSON_ROOT),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
