from __future__ import annotations

import argparse
import json
import subprocess
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
DATABASE_DIR = DATA_DIR / "database"
TABLES_PATH = DATA_DIR / "tables.json"
DEFAULT_SPLIT = "dev"
FULL_DATASET_SIZE = 1034

# HuggingFace dataset for questions + gold SQL (no databases)
HF_DATASET_ID = "xlangai/spider"
HF_PARQUET_URL = (
    "https://huggingface.co/datasets/xlangai/spider/resolve/refs%2Fconvert%2Fparquet"
    "/default/spider/validation-00000-of-00001.parquet"
)

# test-suite-sql-eval repo for EX evaluation code + tables.json
TEST_SUITE_REPO = "https://github.com/taoyds/test-suite-sql-eval.git"
TEST_SUITE_TABLES_URL = (
    "https://raw.githubusercontent.com/taoyds/test-suite-sql-eval/master/tables.json"
)

# Original spider dev databases (from Yale release, mirrored on HF)
# These are the sqlite files needed for EX evaluation.
SPIDER_DEV_ZIP_URL = (
    "https://huggingface.co/datasets/richardr1126/spider-context-instruct"
    "/resolve/main/database.zip"
)

EXTERNAL_SPIDER_DIR = REPO_ROOT / "external" / "spider"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Spider EX-eval dataset manifest.")
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--items", type=int)
    parser.add_argument("--skip-databases", action="store_true",
                        help="Skip downloading sqlite databases (if already present).")
    return parser.parse_args()


def _run(cmd: list[str], *, cwd: Path | None = None) -> None:
    completed = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, check=False)
    if completed.stdout.strip():
        print(completed.stdout.strip())
    if completed.stderr.strip():
        print(completed.stderr.strip())
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed ({completed.returncode}): {' '.join(cmd)}")


def ensure_test_suite_eval() -> None:
    """Clone taoyds/test-suite-sql-eval into external/spider, replacing taoyds/spider if needed."""
    import shutil

    needs_clone = True
    if (EXTERNAL_SPIDER_DIR / ".git").exists():
        # Check if it's already the correct repo
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            cwd=EXTERNAL_SPIDER_DIR,
            capture_output=True,
            text=True,
            check=False,
        )
        current_url = result.stdout.strip().rstrip("/")
        expected_url = TEST_SUITE_REPO.rstrip(".git").rstrip("/")
        if "test-suite-sql-eval" in current_url:
            print(f"[skip] test-suite-sql-eval already cloned at {EXTERNAL_SPIDER_DIR}")
            needs_clone = False
        else:
            print(f"[replace] {current_url} -> {TEST_SUITE_REPO}")
            shutil.rmtree(EXTERNAL_SPIDER_DIR)

    if needs_clone:
        EXTERNAL_SPIDER_DIR.parent.mkdir(parents=True, exist_ok=True)
        print(f"[clone] {TEST_SUITE_REPO} -> {EXTERNAL_SPIDER_DIR}")
        _run(["git", "clone", "--depth", "1", "--filter=blob:none", TEST_SUITE_REPO,
              str(EXTERNAL_SPIDER_DIR)])


def ensure_tables_json() -> None:
    """Download tables.json from test-suite-sql-eval if not already present."""
    if TABLES_PATH.exists():
        print(f"[skip] tables.json already at {TABLES_PATH}")
        return
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    # First try: from the cloned repo
    repo_tables = EXTERNAL_SPIDER_DIR / "tables.json"
    if repo_tables.exists():
        import shutil
        shutil.copy(repo_tables, TABLES_PATH)
        print(f"[copy] tables.json from {repo_tables}")
        return
    # Fallback: download directly
    print(f"[download] tables.json from {TEST_SUITE_TABLES_URL}")
    raw = urllib.request.urlopen(TEST_SUITE_TABLES_URL, timeout=60).read()
    TABLES_PATH.write_bytes(raw)
    print(f"[ok] tables.json -> {TABLES_PATH}")


def ensure_databases(skip: bool) -> None:
    """Download and unpack spider dev databases into data/database/."""
    if DATABASE_DIR.exists() and any(DATABASE_DIR.iterdir()):
        print(f"[skip] databases already present at {DATABASE_DIR}")
        return
    if skip:
        print("[skip] --skip-databases set; skipping database download")
        return
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[download] spider databases from {SPIDER_DEV_ZIP_URL}")
    raw = urllib.request.urlopen(SPIDER_DEV_ZIP_URL, timeout=300).read()
    with zipfile.ZipFile(__import__("io").BytesIO(raw)) as zf:
        # The zip may have a top-level prefix directory; extract database/ entries
        members = [m for m in zf.namelist() if "database/" in m]
        if not members:
            members = zf.namelist()
        for member in members:
            # Normalize path: strip leading prefix up to and including "database/"
            parts = member.replace("\\", "/").split("/")
            try:
                db_idx = next(i for i, p in enumerate(parts) if p == "database")
            except StopIteration:
                continue
            rel = "/".join(parts[db_idx:])
            if not rel or rel.endswith("/"):
                continue
            dest = DATA_DIR / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src:
                dest.write_bytes(src.read())
    print(f"[ok] databases extracted to {DATABASE_DIR}")


def _load_hf_parquet_rows() -> list[dict]:
    """Download xlangai/spider validation parquet and return rows as dicts."""
    try:
        import pyarrow.parquet as pq  # type: ignore
        import io as _io
        raw = urllib.request.urlopen(HF_PARQUET_URL, timeout=120).read()
        table = pq.read_table(_io.BytesIO(raw))
        return table.to_pydict()
    except Exception:
        pass

    # Fallback: datasets library
    try:
        import datasets as hf_datasets  # type: ignore
        ds = hf_datasets.load_dataset(HF_DATASET_ID, split="validation")
        return list(ds)
    except Exception as exc:
        raise RuntimeError(
            f"Unable to load xlangai/spider from HuggingFace. "
            f"Install pyarrow or datasets. Error: {exc}"
        ) from exc


def _load_questions() -> list[dict]:
    """Load dev questions either from HF or from local dev.json fallback."""
    local_dev = DATA_DIR / "dev.json"
    if local_dev.exists():
        rows = json.loads(local_dev.read_text())
        if isinstance(rows, list) and rows:
            print(f"[skip] using existing dev.json ({len(rows)} rows)")
            return rows

    print(f"[download] questions from HuggingFace {HF_DATASET_ID}")
    try:
        col_data = _load_hf_parquet_rows()
        if isinstance(col_data, dict):
            # pyarrow columnar format
            n = len(col_data.get("question", []))
            rows = [
                {k: col_data[k][i] for k in col_data}
                for i in range(n)
            ]
        else:
            rows = col_data
        print(f"[ok] downloaded {len(rows)} questions from HF")
        return rows
    except Exception as exc:
        raise RuntimeError(f"Failed to download Spider questions: {exc}") from exc


def _build_context(*, schema_text: str, db_id: str) -> str:
    sections = [
        f"Database id: {db_id}",
        "Database schema:",
        schema_text.strip(),
    ]
    return "\n".join(s for s in sections if s)


def main() -> None:
    args = parse_args()
    if args.split != DEFAULT_SPLIT:
        raise SystemExit("Only the official Spider dev split is supported.")

    ensure_test_suite_eval()
    ensure_tables_json()
    ensure_databases(skip=bool(args.skip_databases))

    rows = _load_questions()

    tables_by_db: dict[str, dict] = {
        str(entry.get("db_id") or "").strip(): entry
        for entry in json.loads(TABLES_PATH.read_text())
        if isinstance(entry, dict) and str(entry.get("db_id") or "").strip()
    }

    requested_items = int(args.items) if args.items is not None else FULL_DATASET_SIZE
    requested_items = max(1, min(requested_items, FULL_DATASET_SIZE))

    items = []
    for source_index, row in enumerate(rows[:requested_items]):
        if not isinstance(row, dict):
            continue
        db_id = str(row.get("db_id") or "").strip()
        if not db_id:
            raise ValueError(f"Spider row {source_index} is missing db_id.")

        table_entry = tables_by_db.get(db_id)
        if table_entry is None:
            raise ValueError(f"tables.json missing schema for db_id={db_id!r}")

        schema_text = schema_text_from_table_entry(table_entry)
        expected_answer = str(row.get("query") or "").strip()

        # db_path relative to task root for context display; actual path resolved at eval time
        db_rel = f"data/database/{db_id}/{db_id}.sqlite"

        items.append({
            "item_id": f"spider-{args.split}-{source_index}",
            "name": f"Spider {args.split} {source_index + 1}",
            "prompt": str(row.get("question") or "").strip(),
            "context": _build_context(schema_text=schema_text, db_id=db_id),
            "expected_answer": expected_answer,
            "metadata": {
                "dataset": "spider",
                "source_dataset": "xlangai/spider",
                "source_split": DEFAULT_SPLIT,
                "source_index": source_index,
                "db_id": db_id,
                "db_path": db_rel,
                "answer_format": "sql",
                "verifier_style": "spider-execution",
            },
        })

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
