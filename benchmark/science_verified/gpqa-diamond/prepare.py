from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import urllib.request
import zipfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MANIFEST_PATH = DATA_DIR / "questions.json"
DATASET_ID = "gpqa_diamond"
DATASET_ZIP_URL = "https://github.com/idavidrein/gpqa/raw/main/dataset.zip"
DATASET_PASSWORD = "deserted-untie-orchid"
DATASET_MEMBER = "dataset/gpqa_diamond.csv"
FULL_DATASET_SIZE = 198


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and normalize GPQA Diamond into questions.json.")
    parser.add_argument("--items", type=int, default=FULL_DATASET_SIZE)
    return parser.parse_args()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True))
    temp_path.replace(path)


def _load_gpqa_rows() -> list[dict[str, str]]:
    raw_zip = urllib.request.urlopen(DATASET_ZIP_URL, timeout=120).read()
    archive = zipfile.ZipFile(io.BytesIO(raw_zip))
    with archive.open(DATASET_MEMBER, pwd=DATASET_PASSWORD.encode("utf-8")) as handle:
        text_handle = io.TextIOWrapper(handle, encoding="utf-8", newline="")
        return [dict(row) for row in csv.DictReader(text_handle)]


def _shuffle_choices(record_id: str, correct: str, incorrects: list[str]) -> tuple[list[str], int]:
    seed = int(hashlib.sha256(record_id.encode("utf-8")).hexdigest()[:16], 16)
    rng = random.Random(seed)
    choices = [str(correct).strip(), *[str(choice).strip() for choice in incorrects]]
    rng.shuffle(choices)
    return choices, choices.index(str(correct).strip())


def main() -> None:
    args = _parse_args()
    requested_items = max(1, min(int(args.items or FULL_DATASET_SIZE), FULL_DATASET_SIZE))
    rows = _load_gpqa_rows()

    items = []
    for source_index, row in enumerate(rows):
        if source_index >= requested_items:
            break
        record_id = str(row.get("Record ID") or "").strip()
        correct = str(row.get("Correct Answer") or "").strip()
        incorrects = [
            str(row.get("Incorrect Answer 1") or "").strip(),
            str(row.get("Incorrect Answer 2") or "").strip(),
            str(row.get("Incorrect Answer 3") or "").strip(),
        ]
        choices, correct_choice_index = _shuffle_choices(record_id, correct, incorrects)
        items.append(
            {
                "item_id": f"gpqa-diamond-{record_id or source_index}",
                "question_id": record_id or source_index,
                "name": f"GPQA Diamond {str(row.get('High-level domain') or '').strip() or 'science'} {source_index + 1}",
                "prompt": str(row.get("Question") or "").strip(),
                "choices": choices,
                "expected_answer": correct,
                "metadata": {
                    "dataset": "gpqa-diamond",
                    "source_split": "official:diamond",
                    "source_index": source_index,
                    "record_id": record_id,
                    "high_level_domain": str(row.get("High-level domain") or "").strip(),
                    "subdomain": str(row.get("Subdomain") or "").strip(),
                    "correct_choice_index": correct_choice_index,
                    "answer_aliases": [correct],
                },
            }
        )

    manifest = {
        "dataset_id": DATASET_ID,
        "split": "official:diamond",
        "dataset_size": FULL_DATASET_SIZE,
        "prepared_count": len(items),
        "items": items,
    }
    _write_json(MANIFEST_PATH, manifest)
    print(f"Wrote {len(items)} GPQA Diamond items to {MANIFEST_PATH}.")


if __name__ == "__main__":
    main()
