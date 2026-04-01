from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.text2sql_verified.text2sql_support import schema_text_from_table_entry, write_json

DATA_DIR = ROOT / "data"
SOURCE_DIR = DATA_DIR / "source"
DATABASE_DIR = DATA_DIR / "database"
MANIFEST_PATH = DATA_DIR / "questions.json"
DEFAULT_SPLIT = "dev"
FULL_DATASET_SIZE = 2494
OFFICIAL_SOURCE_URL = "https://raw.githubusercontent.com/xjtu-intsoft/chase/page/data/Chase.zip"
OFFICIAL_DATABASE_URL = "https://raw.githubusercontent.com/xjtu-intsoft/chase/page/data/database.zip"
SOURCE_INFO_PATH = SOURCE_DIR / "source_info.json"
DATABASE_INFO_PATH = DATABASE_DIR / "source_info.json"
SOURCE_INFO = {
    "source_repo": "xjtu-intsoft/chase",
    "source_ref": "page",
    "source_archive": OFFICIAL_SOURCE_URL,
}
SOURCE_MEMBERS = {
    "dev.json": "chase_dev.json",
    "train.json": "chase_train.json",
    "test.json": "chase_test.json",
    "tables.json": "chase_tables.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize a local CHASE text-to-SQL manifest prefix.")
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--items", type=int, default=FULL_DATASET_SIZE)
    return parser.parse_args()


def _download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_suffix(destination.suffix + ".tmp")
    try:
        with urllib.request.urlopen(url) as response, temp_path.open("wb") as handle:
            shutil.copyfileobj(response, handle)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        subprocess.run(
            ["curl", "-L", "--fail", "-o", str(temp_path), url],
            check=True,
        )
    temp_path.replace(destination)


def _extract_zip_info(archive: zipfile.ZipFile, info: zipfile.ZipInfo, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_suffix(destination.suffix + ".tmp")
    with archive.open(info) as source, temp_path.open("wb") as target:
        shutil.copyfileobj(source, target)
    temp_path.replace(destination)


def _decode_zip_name(name: str) -> str:
    try:
        return name.encode("cp437").decode("gbk")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return name


def _zip_members_by_name(archive: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    return {_decode_zip_name(info.filename): info for info in archive.infolist()}


def _read_json(path: Path) -> object:
    return json.loads(path.read_text())


def _ensure_local_source_files() -> Path:
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    current_info = _read_json(SOURCE_INFO_PATH) if SOURCE_INFO_PATH.exists() else None
    needs_refresh = current_info != SOURCE_INFO
    for filename in SOURCE_MEMBERS:
        if not (SOURCE_DIR / filename).exists():
            needs_refresh = True
    if not needs_refresh:
        return SOURCE_DIR

    with tempfile.TemporaryDirectory() as temp_dir:
        archive_path = Path(temp_dir) / "Chase.zip"
        print(f"Downloading official CHASE source archive to {archive_path} ...")
        _download_file(OFFICIAL_SOURCE_URL, archive_path)
        with zipfile.ZipFile(archive_path) as archive:
            members = _zip_members_by_name(archive)
            for output_name, member_name in SOURCE_MEMBERS.items():
                info = members.get(member_name)
                if info is None:
                    raise FileNotFoundError(f"CHASE source archive is missing {member_name}.")
                _extract_zip_info(archive, info, SOURCE_DIR / output_name)
    write_json(SOURCE_INFO_PATH, SOURCE_INFO)
    return SOURCE_DIR


def _ensure_local_databases(required_db_ids: set[str]) -> None:
    DATABASE_DIR.mkdir(parents=True, exist_ok=True)
    missing = [
        db_id
        for db_id in sorted(required_db_ids)
        if not (DATABASE_DIR / f"{db_id}.sqlite").exists()
    ]
    if not missing:
        return

    with tempfile.TemporaryDirectory() as temp_dir:
        archive_path = Path(temp_dir) / "database.zip"
        print(f"Downloading official CHASE database archive to {archive_path} ...")
        _download_file(OFFICIAL_DATABASE_URL, archive_path)
        with zipfile.ZipFile(archive_path) as archive:
            members = _zip_members_by_name(archive)
            for db_id in missing:
                info = members.get(f"database/{db_id}.sqlite")
                if info is None:
                    raise FileNotFoundError(f"CHASE database archive is missing database/{db_id}.sqlite.")
                _extract_zip_info(archive, info, DATABASE_DIR / f"{db_id}.sqlite")
    write_json(
        DATABASE_INFO_PATH,
        {
            "source_repo": "xjtu-intsoft/chase",
            "source_ref": "page",
            "source_archive": OFFICIAL_DATABASE_URL,
        },
    )


def _build_context(*, db_id: str, schema_text: str, history: list[str]) -> str:
    sections = [
        f"Database ID: {db_id}",
        schema_text.strip(),
        f"Local SQLite path relative to the task directory: data/database/{db_id}.sqlite",
    ]
    if history:
        history_lines = [f"{index + 1}. {text}" for index, text in enumerate(history)]
        sections.append("Conversation history:\n" + "\n".join(history_lines))
    return "\n\n".join(section for section in sections if section.strip())


def main() -> None:
    args = parse_args()
    requested_items = max(1, min(int(args.items or FULL_DATASET_SIZE), FULL_DATASET_SIZE))

    source_root = _ensure_local_source_files()
    split_path = source_root / f"{args.split}.json"
    tables_path = source_root / "tables.json"
    if not split_path.exists():
        raise FileNotFoundError(f"CHASE source split not found: {split_path}")
    if not tables_path.exists():
        raise FileNotFoundError(f"CHASE schema file not found: {tables_path}")

    interactions = json.loads(split_path.read_text())
    tables = json.loads(tables_path.read_text())
    tables_by_db = {str(entry["db_id"]): entry for entry in tables}

    items = []
    required_db_ids: set[str] = set()
    for interaction_index, row in enumerate(interactions):
        db_id = str(row["database_id"])
        schema_text = schema_text_from_table_entry(tables_by_db[db_id])
        history: list[str] = []
        for turn_index, turn in enumerate(row["interaction"]):
            if len(items) >= requested_items:
                break
            required_db_ids.add(db_id)
            items.append(
                {
                    "item_id": f"chase-{args.split}-{interaction_index}-{turn_index}",
                    "name": f"CHASE {args.split} {interaction_index + 1}.{turn_index + 1}",
                    "prompt": str(turn["utterance"]).strip(),
                    "context": _build_context(db_id=db_id, schema_text=schema_text, history=history),
                    "expected_answer": str(turn["query"]).strip(),
                    "metadata": {
                        "dataset": "chase",
                        "source_repo": "xjtu-intsoft/chase",
                        "source_ref": "page",
                        "source_split": args.split,
                        "interaction_index": interaction_index,
                        "turn_index": turn_index,
                        "db_id": db_id,
                        "db_path": str(Path("data") / "database" / f"{db_id}.sqlite"),
                        "answer_format": "sql",
                        "verifier_style": "spider-execution",
                    },
                }
            )
            history.append(str(turn["utterance"]).strip())
        if len(items) >= requested_items:
            break

    _ensure_local_databases(required_db_ids)

    manifest = {
        "dataset_id": "chase",
        "split": args.split,
        "dataset_size": FULL_DATASET_SIZE,
        "prepared_count": len(items),
        "items": items,
    }
    write_json(MANIFEST_PATH, manifest)
    print(f"Wrote {len(items)} CHASE items to {MANIFEST_PATH}.")


if __name__ == "__main__":
    main()
