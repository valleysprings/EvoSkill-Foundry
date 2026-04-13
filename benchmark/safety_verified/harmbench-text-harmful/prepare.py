from __future__ import annotations

import argparse
import csv
import io
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[2]
TRACK_ROOT = ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(TRACK_ROOT) not in sys.path:
    sys.path.insert(0, str(TRACK_ROOT))

from app.bench.safety_support import write_json
from sync_external import ensure_repo_checkout


DATA_DIR = ROOT / "data"
SOURCE_DIR = DATA_DIR / "source"
MANIFEST_PATH = DATA_DIR / "questions.json"
SOURCE_INFO_PATH = DATA_DIR / "source_info.json"
BEHAVIORS_FILENAME = "harmbench_behaviors_text_test.csv"
ATTACKS_FILENAME = "harmbench_selected_attack_prompts.json"
FULL_DATASET_SIZE = 240
MAX_ATTACKS_PER_BEHAVIOR = 3


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize the local HarmBench harmful text manifest.")
    parser.add_argument("--items", type=int, default=FULL_DATASET_SIZE, help="Optional prefix length.")
    return parser.parse_args()


def _normalize_text(value: object) -> str:
    return " ".join(str(value or "").split()).strip().lower()


def _is_hash_check_row(row: dict[str, Any]) -> bool:
    semantic = _normalize_text(row.get("SemanticCategory"))
    tags = _normalize_text(row.get("Tags"))
    behavior_id = _normalize_text(row.get("BehaviorID"))
    behavior = _normalize_text(row.get("Behavior"))
    return "hash_check" in semantic or "hash_check" in tags or "hash_check" in behavior_id or "hash check" in behavior


def _load_behavior_rows() -> list[dict[str, str]]:
    source_root = ensure_repo_checkout("harmbench")
    csv_path = source_root / "data" / "behavior_datasets" / BEHAVIORS_FILENAME
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing HarmBench behaviors CSV: {csv_path}")
    raw_text = csv_path.read_text()
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    (SOURCE_DIR / BEHAVIORS_FILENAME).write_text(raw_text)
    rows = [dict(row) for row in csv.DictReader(io.StringIO(raw_text))]
    filtered = [row for row in rows if not _is_hash_check_row(row)]
    return filtered


def _load_human_jailbreak_templates() -> list[str]:
    source_root = ensure_repo_checkout("harmbench")
    module_path = source_root / "baselines" / "human_jailbreaks" / "jailbreaks.py"
    spec = importlib.util.spec_from_file_location("harmbench_human_jailbreaks", module_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Unable to load HarmBench human jailbreaks from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    jailbreaks = getattr(module, "JAILBREAKS", None)
    if not isinstance(jailbreaks, list) or not jailbreaks:
        raise ValueError("HarmBench human jailbreak baseline did not expose a non-empty JAILBREAKS list.")
    return [str(item).strip() for item in jailbreaks if str(item).strip()]


def build_questions(
    rows: list[dict[str, str]],
    *,
    jailbreak_templates: list[str],
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    selected_prompt_rows: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows, start=1):
        behavior_id = str(row.get("BehaviorID") or f"harmbench-{row_index:04d}").strip()
        behavior = str(row.get("Behavior") or "").strip()
        context = str(row.get("ContextString") or "").strip()
        selected: list[dict[str, Any]] = []
        behavior_payload = f"{context}\n\n---\n\n{behavior}" if context else behavior
        for template_index, template in enumerate(jailbreak_templates[:MAX_ATTACKS_PER_BEHAVIOR], start=1):
            selected.append(
                {
                    "attack_id": f"human-jailbreak-{template_index}",
                    "name": f"human-jailbreak-{template_index}",
                    "prompt": template.format(behavior_payload),
                    "source": "harmbench:human_jailbreaks",
                }
            )
        selected_prompt_rows.append({"behavior_id": behavior_id, "attack_prompts": selected})
        items.append(
            {
                "item_id": f"harmbench-{behavior_id}",
                "name": f"HarmBench {behavior_id}",
                "prompt": behavior,
                "context": context,
                "metadata": {
                    "benchmark": "harmbench",
                    "behavior_id": behavior_id,
                    "behavior_kind": "contextual" if context else "standard",
                    "functional_category": str(row.get("FunctionalCategory") or "").strip(),
                    "semantic_category": str(row.get("SemanticCategory") or "").strip(),
                    "tags": str(row.get("Tags") or "").strip(),
                    "attack_prompts": selected,
                },
            }
        )
    write_json(SOURCE_DIR / ATTACKS_FILENAME, selected_prompt_rows)
    return items


def main() -> None:
    args = _parse_args()
    all_rows = _load_behavior_rows()
    jailbreak_templates = _load_human_jailbreak_templates()
    items = build_questions(all_rows, jailbreak_templates=jailbreak_templates)
    requested_items = max(1, min(int(args.items or len(items)), len(items)))
    prepared_items = items[:requested_items]
    manifest = {
        "dataset_id": "harmbench_text_harmful",
        "split": "text_test:standard+contextual-harmful",
        "dataset_size": len(items),
        "prepared_count": len(prepared_items),
        "items": prepared_items,
    }
    source_info = {
        "benchmark": "harmbench",
        "sources": {
            "behaviors_repo": "https://github.com/centerforaisafety/HarmBench",
            "behaviors_csv": BEHAVIORS_FILENAME,
            "official_human_jailbreaks_module": "baselines/human_jailbreaks/jailbreaks.py",
        },
        "filters": {
            "excluded_rows": "hash_check / copyright-eval rows",
            "included_behavior_kinds": ["standard", "contextual"],
        },
        "fixed_prompt_source": "official_human_jailbreaks",
        "attack_prompts_per_behavior": MAX_ATTACKS_PER_BEHAVIOR,
        "prepared_count": len(prepared_items),
        "dataset_size": len(items),
    }
    write_json(MANIFEST_PATH, manifest)
    write_json(SOURCE_INFO_PATH, source_info)
    print(f"Wrote {len(prepared_items)} HarmBench harmful behaviors to {MANIFEST_PATH}.")


if __name__ == "__main__":
    main()
