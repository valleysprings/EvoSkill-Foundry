from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "benchmark" / "text2sql_verified" / "spider" / "prepare.py"


def _load_prepare_module():
    spec = importlib.util.spec_from_file_location("spider_prepare", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SpiderPrepareTest(unittest.TestCase):
    def test_prepare_uses_local_official_dev_assets(self) -> None:
        module = _load_prepare_module()
        with tempfile.TemporaryDirectory() as tmp_dir:
            task_root = Path(tmp_dir)
            data_dir = task_root / "data"
            db_dir = data_dir / "database" / "concert_singer"
            db_dir.mkdir(parents=True, exist_ok=True)
            (db_dir / "concert_singer.sqlite").write_bytes(b"")

            (data_dir / "dev.json").write_text(
                json.dumps(
                    [
                        {
                            "db_id": "concert_singer",
                            "question": "List singer names.",
                            "query": "SELECT name FROM singer",
                            "query_toks": [],
                            "query_toks_no_value": [],
                            "question_toks": [],
                            "sql": {},
                        }
                    ]
                )
            )
            (data_dir / "tables.json").write_text(
                json.dumps(
                    [
                        {
                            "db_id": "concert_singer",
                            "table_names": ["singer"],
                            "table_names_original": ["singer"],
                            "column_names": [[-1, "*"], [0, "name"]],
                            "column_names_original": [[-1, "*"], [0, "name"]],
                            "column_types": ["text", "text"],
                            "primary_keys": [],
                            "foreign_keys": [],
                        }
                    ]
                )
            )

            module.ROOT = task_root
            module.DATA_DIR = data_dir
            module.MANIFEST_PATH = data_dir / "questions.json"

            with patch.object(sys, "argv", ["prepare.py", "--items", "1"]):
                module.main()

            payload = json.loads((data_dir / "questions.json").read_text())
            self.assertEqual(payload["split"], "dev")
            self.assertEqual(payload["dataset_size"], 1034)
            self.assertEqual(payload["prepared_count"], 1)
            self.assertEqual(len(payload["items"]), 1)
            self.assertEqual(
                payload["items"][0]["metadata"]["db_path"],
                "data/database/concert_singer/concert_singer.sqlite",
            )
            self.assertIn("Database schema:", payload["items"][0]["context"])


if __name__ == "__main__":
    unittest.main()
