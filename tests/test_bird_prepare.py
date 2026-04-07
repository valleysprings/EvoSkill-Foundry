from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "benchmark" / "text2sql_verified" / "bird" / "prepare.py"


def _load_prepare_module():
    spec = importlib.util.spec_from_file_location("bird_prepare", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BirdPrepareTest(unittest.TestCase):
    def test_prepare_uses_local_assets_without_download(self) -> None:
        module = _load_prepare_module()
        with tempfile.TemporaryDirectory() as tmp_dir:
            task_root = Path(tmp_dir)
            data_dir = task_root / "data"
            db_dir = data_dir / "dev_databases" / "sample"
            db_dir.mkdir(parents=True, exist_ok=True)
            (db_dir / "sample.sqlite").write_bytes(b"")

            (data_dir / "dev.json").write_text(
                json.dumps(
                    [
                        {
                            "question_id": 0,
                            "db_id": "sample",
                            "question": "List names.",
                            "SQL": "SELECT name FROM singer",
                            "evidence": "",
                            "difficulty": "simple",
                        }
                    ]
                )
            )
            (data_dir / "dev_tables.json").write_text(
                json.dumps(
                    [
                        {
                            "db_id": "sample",
                            "table_names": ["singer"],
                            "table_names_original": ["singer"],
                            "column_names": [[-1, "*"], [0, "id"], [0, "name"]],
                            "column_names_original": [[-1, "*"], [0, "id"], [0, "name"]],
                            "column_types": ["text", "number", "text"],
                            "primary_keys": [[1, 2]],
                            "foreign_keys": [],
                        }
                    ]
                )
            )

            module.ROOT = task_root
            module.DATA_DIR = data_dir
            module.MANIFEST_PATH = data_dir / "questions.json"

            with (
                patch.object(sys, "argv", ["prepare.py", "--items", "1"]),
                patch.object(module, "_download_file", side_effect=AssertionError("download should not be called")),
            ):
                module.main()

            payload = json.loads((data_dir / "questions.json").read_text())
            self.assertEqual(payload["split"], "dev")
            self.assertEqual(payload["dataset_size"], 1534)
            self.assertEqual(payload["prepared_count"], 1)
            self.assertEqual(len(payload["items"]), 1)
            self.assertEqual(
                payload["items"][0]["metadata"]["db_path"],
                "data/dev_databases/sample/sample.sqlite",
            )
            self.assertIn("id [ NUMBER ] primary_key", payload["items"][0]["context"])
            self.assertIn("name [ TEXT ] primary_key", payload["items"][0]["context"])


if __name__ == "__main__":
    unittest.main()
