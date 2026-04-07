from __future__ import annotations

import argparse
import json
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.text2sql_verified.text2sql_support import schema_text_from_table_entry, write_json

DATA_DIR = ROOT / "data"
MANIFEST_PATH = DATA_DIR / "questions.json"
OFFICIAL_DEV_URL = "https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip"
DEFAULT_SPLIT = "dev"
FULL_DATASET_SIZE = 1534
OUTER_PREFIX = "dev_20240627"
DEV_JSON_NAME = "dev.json"
DEV_TABLES_NAME = "dev_tables.json"
DEV_SQL_NAME = "dev.sql"
DEV_TIED_APPEND_NAME = "dev_tied_append.json"
DEV_DATABASES_DIRNAME = "dev_databases"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize a local official BIRD text-to-SQL manifest prefix.")
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--items", type=int, default=FULL_DATASET_SIZE)
    return parser.parse_args()


def load_existing_manifest() -> dict | None:
    if not MANIFEST_PATH.exists():
        return None
    payload = json.loads(MANIFEST_PATH.read_text())
    return payload if isinstance(payload, dict) else None


def _download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_suffix(destination.suffix + ".tmp")
    with urllib.request.urlopen(url) as response, temp_path.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    temp_path.replace(destination)


def _extract_bytes(archive: zipfile.ZipFile, member_name: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_suffix(destination.suffix + ".tmp")
    with archive.open(member_name) as source, temp_path.open("wb") as target:
        shutil.copyfileobj(source, target)
    temp_path.replace(destination)


def _extract_nested_member(archive: zipfile.ZipFile, member_name: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_suffix(destination.suffix + ".tmp")
    with archive.open(member_name) as source, temp_path.open("wb") as target:
        shutil.copyfileobj(source, target)
    temp_path.replace(destination)


def _should_skip_nested_member(name: str) -> bool:
    parts = [part for part in Path(name).parts if part]
    return (
        not parts
        or parts[0] != DEV_DATABASES_DIRNAME
        or "__MACOSX" in parts
        or any(part.startswith("._") for part in parts)
        or parts[-1] == ".DS_Store"
    )


def _ensure_official_assets(required_db_ids: set[str]) -> None:
    dev_json_path = DATA_DIR / DEV_JSON_NAME
    dev_tables_path = DATA_DIR / DEV_TABLES_NAME
    if dev_json_path.exists() and dev_tables_path.exists():
        missing = [
            db_id
            for db_id in sorted(required_db_ids)
            if not (DATA_DIR / DEV_DATABASES_DIRNAME / db_id / f"{db_id}.sqlite").exists()
        ]
        if not missing:
            return
    archive_path = DATA_DIR / "official_dev.zip"
    if not archive_path.exists():
        print(f"Downloading official BIRD dev package to {archive_path} ...")
        _download_file(OFFICIAL_DEV_URL, archive_path)

    with zipfile.ZipFile(archive_path) as outer_archive:
        for inner_name, output_name in (
            (f"{OUTER_PREFIX}/{DEV_JSON_NAME}", DEV_JSON_NAME),
            (f"{OUTER_PREFIX}/{DEV_TABLES_NAME}", DEV_TABLES_NAME),
            (f"{OUTER_PREFIX}/{DEV_SQL_NAME}", DEV_SQL_NAME),
            (f"{OUTER_PREFIX}/{DEV_TIED_APPEND_NAME}", DEV_TIED_APPEND_NAME),
        ):
            destination = DATA_DIR / output_name
            if not destination.exists():
                _extract_bytes(outer_archive, inner_name, destination)

        nested_zip_path = DATA_DIR / f"{DEV_DATABASES_DIRNAME}.zip"
        if not nested_zip_path.exists():
            _extract_bytes(outer_archive, f"{OUTER_PREFIX}/{DEV_DATABASES_DIRNAME}.zip", nested_zip_path)

    with zipfile.ZipFile(DATA_DIR / f"{DEV_DATABASES_DIRNAME}.zip") as db_archive:
        for member_name in db_archive.namelist():
            if _should_skip_nested_member(member_name):
                continue
            relative_path = Path(member_name)
            db_id = relative_path.parts[1] if len(relative_path.parts) > 1 else ""
            if required_db_ids and db_id not in required_db_ids:
                continue
            destination = DATA_DIR / relative_path
            if member_name.endswith("/"):
                destination.mkdir(parents=True, exist_ok=True)
                continue
            if destination.exists():
                continue
            _extract_nested_member(db_archive, member_name, destination)


def _load_json(path: Path) -> object:
    return json.loads(path.read_text())


def _build_context(*, schema_text: str, evidence: str, db_id: str) -> str:
    sections = [
        f"Database id: {db_id}",
        "Database schema:",
        schema_text.strip(),
        (
            "Local SQLite path relative to the task directory: "
            f"data/{DEV_DATABASES_DIRNAME}/{db_id}/{db_id}.sqlite"
        ),
        (
            "Local database description directory relative to the task directory: "
            f"data/{DEV_DATABASES_DIRNAME}/{db_id}/database_description"
        ),
    ]
    evidence_text = evidence.strip()
    if evidence_text:
        sections.extend(["Evidence:", evidence_text])
    return "\n".join(section for section in sections if section)


def main() -> None:
    args = parse_args()
    if args.split != DEFAULT_SPLIT:
        raise SystemExit("Only the official BIRD dev split is supported by this prepare script.")

    requested_items = max(1, min(int(args.items or FULL_DATASET_SIZE), FULL_DATASET_SIZE))
    existing = load_existing_manifest()
    existing_count = int(existing.get("prepared_count") or len(existing.get("items") or [])) if existing else 0

    try:
        dev_json_path = DATA_DIR / DEV_JSON_NAME
        dev_tables_path = DATA_DIR / DEV_TABLES_NAME
        if dev_json_path.exists() and dev_tables_path.exists():
            rows = _load_json(dev_json_path)
            table_entries = _load_json(dev_tables_path)
        else:
            archive_path = DATA_DIR / "official_dev.zip"
            if not archive_path.exists():
                _download_file(OFFICIAL_DEV_URL, archive_path)
            with zipfile.ZipFile(archive_path) as outer_archive:
                rows = json.loads(outer_archive.read(f"{OUTER_PREFIX}/{DEV_JSON_NAME}"))
                table_entries = json.loads(outer_archive.read(f"{OUTER_PREFIX}/{DEV_TABLES_NAME}"))
    except Exception as exc:
        if existing_count > 0:
            print(
                f"Unable to refresh official BIRD assets ({exc}); keeping existing manifest "
                f"with {existing_count} items at {MANIFEST_PATH}."
            )
            return
        raise

    if not isinstance(rows, list) or not isinstance(table_entries, list):
        raise ValueError("Official BIRD package is malformed: dev.json/dev_tables.json must be lists.")

    selected_rows = rows[:requested_items]
    required_db_ids = {str(row.get("db_id") or "").strip() for row in selected_rows if str(row.get("db_id") or "").strip()}
    _ensure_official_assets(required_db_ids)

    tables_by_db = {
        str(entry.get("db_id") or "").strip(): entry
        for entry in table_entries
        if isinstance(entry, dict) and str(entry.get("db_id") or "").strip()
    }

    items = []
    for source_index, row in enumerate(selected_rows):
        if not isinstance(row, dict):
            continue
        db_id = str(row.get("db_id") or "").strip()
        if not db_id:
            raise ValueError(f"BIRD row {source_index} is missing db_id.")
        table_entry = tables_by_db.get(db_id)
        if table_entry is None:
            raise ValueError(f"BIRD dev_tables.json is missing schema for db_id={db_id!r}.")
        schema_text = schema_text_from_table_entry(table_entry)
        evidence = str(row.get("evidence") or "").strip()
        db_relative_path = Path("data") / DEV_DATABASES_DIRNAME / db_id / f"{db_id}.sqlite"
        description_relative_path = Path("data") / DEV_DATABASES_DIRNAME / db_id / "database_description"
        items.append(
            {
                "item_id": f"bird-{args.split}-{source_index}",
                "name": f"BIRD {args.split} {source_index + 1}",
                "prompt": str(row.get("question") or "").strip(),
                "context": _build_context(schema_text=schema_text, evidence=evidence, db_id=db_id),
                "expected_answer": str(row.get("SQL") or "").strip(),
                "metadata": {
                    "dataset": "bird",
                    "source_dataset": "AlibabaResearch/DAMO-ConvAI BIRD official",
                    "source_split": args.split,
                    "source_index": source_index,
                    "source_question_id": row.get("question_id"),
                    "db_id": db_id,
                    "db_path": str(db_relative_path),
                    "database_description_path": str(description_relative_path),
                    "difficulty": str(row.get("difficulty") or "").strip(),
                    "evidence": evidence,
                    "answer_format": "sql",
                    "verifier_style": "bird-execution",
                },
            }
        )

    manifest = {
        "dataset_id": "bird",
        "split": args.split,
        "dataset_size": FULL_DATASET_SIZE,
        "prepared_count": len(items),
        "items": items,
    }
    write_json(MANIFEST_PATH, manifest)
    print(f"Wrote {len(items)} BIRD items to {MANIFEST_PATH}.")


if __name__ == "__main__":
    main()
