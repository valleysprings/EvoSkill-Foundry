from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.text2sql_verified.text2sql_support import schema_text_from_table_entry, write_json

DATA_DIR = ROOT / "data"
MANIFEST_PATH = DATA_DIR / "questions.json"
DEFAULT_SPLIT = "dev"
FULL_DATASET_SIZE = 1034
DEV_JSON_NAME = "dev.json"
DEV_TABLES_NAME = "tables.json"
DEV_DATABASES_DIRNAME = "database"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize a local official Spider text-to-SQL manifest prefix.")
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--items", type=int)
    return parser.parse_args()


def load_existing_manifest() -> dict | None:
    if not MANIFEST_PATH.exists():
        return None
    payload = json.loads(MANIFEST_PATH.read_text())
    return payload if isinstance(payload, dict) else None


def _load_json(path: Path) -> object:
    return json.loads(path.read_text())


def _resolve_db_relative_path(db_dir_name: str, db_id: str) -> Path:
    db_dir = DATA_DIR / db_dir_name / db_id
    exact_path = db_dir / f"{db_id}.sqlite"
    if exact_path.exists():
        return exact_path.relative_to(ROOT)

    sqlite_candidates = sorted(path for path in db_dir.glob("*.sqlite") if path.is_file())
    if sqlite_candidates:
        return sqlite_candidates[0].relative_to(ROOT)

    raise FileNotFoundError(f"Spider sqlite database not found for db_id={db_id!r} under {db_dir}")


def _build_context(*, schema_text: str, db_id: str, db_relative_path: Path) -> str:
    sections = [
        f"Database id: {db_id}",
        "Database schema:",
        schema_text.strip(),
        f"Local SQLite path relative to the task directory: {db_relative_path}",
    ]
    return "\n".join(section for section in sections if section)


def main() -> None:
    args = parse_args()
    if args.split != DEFAULT_SPLIT:
        raise SystemExit("Only the official Spider dev split is supported by this prepare script.")

    questions_path = DATA_DIR / DEV_JSON_NAME
    tables_path = DATA_DIR / DEV_TABLES_NAME
    requested_items = int(args.items) if args.items is not None else FULL_DATASET_SIZE
    requested_items = max(1, min(requested_items, FULL_DATASET_SIZE))

    existing = load_existing_manifest()
    existing_count = int(existing.get("prepared_count") or len(existing.get("items") or [])) if existing else 0

    try:
        rows = _load_json(questions_path)
        table_entries = _load_json(tables_path)
    except Exception as exc:
        if existing_count > 0:
            print(
                f"Unable to refresh Spider from local official assets ({exc}); "
                f"keeping existing manifest with {existing_count} items at {MANIFEST_PATH}."
            )
            return
        raise

    if not isinstance(rows, list) or not isinstance(table_entries, list):
        raise ValueError("Official Spider assets are malformed: question and table files must be lists.")

    tables_by_db = {
        str(entry.get("db_id") or "").strip(): entry
        for entry in table_entries
        if isinstance(entry, dict) and str(entry.get("db_id") or "").strip()
    }

    items = []
    for source_index, row in enumerate(rows[:requested_items]):
        if not isinstance(row, dict):
            continue
        db_id = str(row.get("db_id") or "").strip()
        if not db_id:
            raise ValueError(f"Spider row {source_index} is missing db_id.")

        table_entry = tables_by_db.get(db_id)
        if table_entry is None:
            raise ValueError(f"Spider tables metadata is missing schema for db_id={db_id!r}.")

        db_relative_path = _resolve_db_relative_path(DEV_DATABASES_DIRNAME, db_id)
        schema_text = schema_text_from_table_entry(table_entry)
        expected_answer = str(row.get("query") or "").strip()
        items.append(
            {
                "item_id": f"spider-{args.split}-{source_index}",
                "name": f"Spider {args.split} {source_index + 1}",
                "prompt": str(row.get("question") or "").strip(),
                "context": _build_context(
                    schema_text=schema_text,
                    db_id=db_id,
                    db_relative_path=db_relative_path,
                ),
                "expected_answer": expected_answer,
                "metadata": {
                    "dataset": "spider",
                    "source_dataset": "Spider official",
                    "source_split": DEFAULT_SPLIT,
                    "source_index": source_index,
                    "db_id": db_id,
                    "db_path": str(db_relative_path),
                    "answer_format": "sql",
                    "verifier_style": "adapter-normalized-sql",
                },
            }
        )

    manifest = {
        "dataset_id": "spider",
        "split": DEFAULT_SPLIT,
        "dataset_size": FULL_DATASET_SIZE,
        "prepared_count": len(items),
        "items": items,
    }
    write_json(MANIFEST_PATH, manifest)
    print(f"Wrote {len(items)} Spider items to {MANIFEST_PATH}.")


if __name__ == "__main__":
    main()
