from __future__ import annotations

import sqlite3
import tempfile
import textwrap
import unittest
from pathlib import Path

from benchmark.text2sql_verified.text2sql_support import (
    evaluate_bird_execution_candidate,
    evaluate_chase_execution_candidate,
    normalize_sql_text,
    parse_bird_prompt,
)


class TextToSqlSupportTest(unittest.TestCase):
    def test_normalize_sql_text_extracts_code_block(self) -> None:
        actual = "```sql\nSELECT  *  FROM singer ;\n```"
        self.assertEqual(normalize_sql_text(actual), "select * from singer")

    def test_normalize_sql_text_preserves_quoted_literals(self) -> None:
        actual = 'SELECT Name FROM singer WHERE Country = "France";'
        expected = 'select name from singer where country = "France"'
        self.assertEqual(normalize_sql_text(actual), expected)

    def test_parse_bird_prompt_splits_schema_and_question(self) -> None:
        schema, question = parse_bird_prompt(
            "[INST] Here is a database schema:\nfoo :\nid [ INTEGER ]\n\n"
            "Please write me a SQL statement that answers the following question: how many rows? [/INST]"
        )
        self.assertIn("foo :", schema)
        self.assertEqual(question, "how many rows?")

    def test_bird_execution_evaluation_accepts_result_equivalent_sql(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            db_path = root / "sample.sqlite"
            connection = sqlite3.connect(db_path)
            try:
                cursor = connection.cursor()
                cursor.execute("CREATE TABLE scores(value INTEGER)")
                cursor.executemany("INSERT INTO scores(value) VALUES (?)", [(1,), (5,), (3,)])
                connection.commit()
            finally:
                connection.close()

            candidate_path = root / "candidate.py"
            candidate_path.write_text(
                textwrap.dedent(
                    """
                    def solve(question: dict) -> str:
                        return "SELECT MAX(value) FROM scores"
                    """
                ).strip()
                + "\n"
            )

            result = evaluate_bird_execution_candidate(
                task={
                    "entry_symbol": "solve",
                    "task_dir": str(root),
                    "question_item": {
                        "item_id": "bird-dev-0",
                        "name": "BIRD dev 1",
                        "expected_answer": "SELECT value FROM scores ORDER BY value DESC LIMIT 1",
                        "raw_expected_answer": "SELECT value FROM scores ORDER BY value DESC LIMIT 1",
                        "metadata": {
                            "db_id": "sample",
                            "db_path": str(db_path),
                        },
                    },
                },
                candidate_path=candidate_path,
            )

        self.assertEqual(result["verifier_status"], "pass")
        self.assertEqual(result["objective"], 1.0)

    def test_bird_execution_evaluation_marks_invalid_sql_as_fail(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            db_path = root / "sample.sqlite"
            connection = sqlite3.connect(db_path)
            try:
                cursor = connection.cursor()
                cursor.execute("CREATE TABLE scores(value INTEGER)")
                cursor.executemany("INSERT INTO scores(value) VALUES (?)", [(1,), (5,), (3,)])
                connection.commit()
            finally:
                connection.close()

            candidate_path = root / "candidate.py"
            candidate_path.write_text(
                textwrap.dedent(
                    """
                    def solve(question: dict) -> str:
                        return "SELECT missing_column FROM scores"
                    """
                ).strip()
                + "\n"
            )

            result = evaluate_bird_execution_candidate(
                task={
                    "entry_symbol": "solve",
                    "task_dir": str(root),
                    "question_item": {
                        "item_id": "bird-dev-0",
                        "name": "BIRD dev 1",
                        "expected_answer": "SELECT MAX(value) FROM scores",
                        "raw_expected_answer": "SELECT MAX(value) FROM scores",
                        "metadata": {
                            "db_id": "sample",
                            "db_path": str(db_path),
                        },
                    },
                },
                candidate_path=candidate_path,
            )

        self.assertEqual(result["verifier_status"], "fail")
        self.assertIn("execution_error", result["test_results"][0])

    def test_chase_execution_evaluation_accepts_result_equivalent_sql(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        chase_root = repo_root / "benchmark" / "text2sql_verified" / "chase"
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            db_path = root / "sample.sqlite"
            connection = sqlite3.connect(db_path)
            try:
                cursor = connection.cursor()
                cursor.execute("CREATE TABLE scores(value INTEGER)")
                cursor.executemany("INSERT INTO scores(value) VALUES (?)", [(1,), (5,), (3,)])
                connection.commit()
            finally:
                connection.close()

            candidate_path = Path(temp_dir) / "candidate.py"
            candidate_path.write_text(
                textwrap.dedent(
                    """
                    def solve(question: dict) -> str:
                        return "SELECT MAX(value) FROM scores"
                    """
                ).strip()
                + "\n"
            )

            result = evaluate_chase_execution_candidate(
                task={
                    "entry_symbol": "solve",
                    "task_dir": str(chase_root),
                    "question_item": {
                        "item_id": "chase-dev-0-0",
                        "name": "CHASE dev 1.1",
                        "expected_answer": "SELECT value FROM scores ORDER BY value DESC LIMIT 1",
                        "raw_expected_answer": "SELECT value FROM scores ORDER BY value DESC LIMIT 1",
                        "metadata": {
                            "db_id": "sample",
                            "db_path": str(db_path),
                            "source_split": "dev",
                            "interaction_index": 0,
                            "turn_index": 0,
                        },
                    },
                },
                candidate_path=candidate_path,
            )

        self.assertEqual(result["verifier_status"], "pass")
        self.assertEqual(result["objective"], 1.0)


if __name__ == "__main__":
    unittest.main()
