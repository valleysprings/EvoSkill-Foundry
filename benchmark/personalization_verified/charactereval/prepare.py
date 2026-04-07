from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TRACK_ROOT = ROOT.parent
REPO_ROOT = ROOT.parents[2]
for candidate in (TRACK_ROOT, REPO_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from sync_external import ensure_repo_checkout
from app.bench.personalization_support import benchmark_metadata, write_manifest


MANIFEST_PATH = ROOT / "data" / "questions.json"
DATASET_ID = "charactereval_test_data"
SPLIT = "official:test_data"
FULL_DATASET_SIZE = 4564
PROMPT = "Write the next assistant turn only. Stay faithful to the character profile and the dialogue context."


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize the public CharacterEval test data.")
    parser.add_argument("--items", type=int, default=None, help="Optional prefix length.")
    return parser.parse_args(argv)


def _build_items() -> list[dict[str, object]]:
    source_root = ensure_repo_checkout("charactereval")
    rows = json.loads((source_root / "data" / "test_data.jsonl").read_text())
    if not isinstance(rows, list):
        raise ValueError("CharacterEval test_data.jsonl must contain a JSON list.")
    character_profiles = json.loads((source_root / "data" / "character_profiles.json").read_text())
    metric_map = json.loads((source_root / "data" / "id2metric.jsonl").read_text())
    if not isinstance(character_profiles, dict) or not isinstance(metric_map, dict):
        raise ValueError("CharacterEval auxiliary files must contain JSON objects.")

    items: list[dict[str, object]] = []
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            continue
        source_id = str(row.get("id") or "").strip()
        role = str(row.get("role") or "").strip()
        context = str(row.get("context") or "").strip()
        novel_name = str(row.get("novel_name") or "").strip()
        if not source_id or not role or not context:
            continue
        profile = character_profiles.get(role)
        metric_pairs = metric_map.get(source_id) or []
        metrics = [str(pair[0]).strip() for pair in metric_pairs if isinstance(pair, list) and pair]
        items.append(
            {
                "item_id": f"charactereval-{source_id}",
                "name": f"CharacterEval {role} #{index}",
                "prompt": PROMPT,
                "context": "\n\n".join(
                    block
                    for block in (
                        f"Character: {role}",
                        f"Source work: {novel_name}" if novel_name else "",
                        f"Character profile:\n{json.dumps(profile, ensure_ascii=False, indent=2)}" if profile is not None else "",
                        f"Dialogue context:\n{context}",
                        f"Judged metrics: {', '.join(metrics)}" if metrics else "",
                    )
                    if block
                ),
                "expected_answer": "__judge_model__",
                "metadata": benchmark_metadata(
                    benchmark="charactereval",
                    benchmark_category="explicit_character_persona",
                    interaction_mode="multi_turn",
                    task_shape="dialogue_judgement",
                    scoring_mode="judge_model",
                    extra={
                        "source_id": source_id,
                        "character_name": role,
                        "novel_name": novel_name,
                        "charactereval_metrics": metrics,
                    },
                ),
            }
        )
    if len(items) != FULL_DATASET_SIZE:
        raise ValueError(f"Expected {FULL_DATASET_SIZE} CharacterEval rows, found {len(items)}.")
    return items


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    items = _build_items()
    requested = len(items) if args.items is None else max(1, min(int(args.items), len(items)))
    write_manifest(
        MANIFEST_PATH,
        dataset_id=DATASET_ID,
        split=SPLIT,
        items=items[:requested],
        dataset_size=FULL_DATASET_SIZE,
    )
    print(f"Wrote {requested} CharacterEval rows to {MANIFEST_PATH}.")


if __name__ == "__main__":
    main()
