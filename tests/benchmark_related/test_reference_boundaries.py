from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BANNED_PATTERNS = (
    re.compile(r'["\']external/'),
    re.compile(r'["\']External/'),
    re.compile(r'/\s*["\']external["\']'),
    re.compile(r'/\s*["\']External["\']'),
)
ALLOWED_EXTERNAL_REFERENCE_PATHS = {
    "benchmark/agent_verified/alfworld/prepare.py",
    "benchmark/agent_verified/alfworld/verifier.py",
    "benchmark/personalization_verified/sync_external.py",
    "benchmark/safety_verified/sync_external.py",
}


class ReferenceBoundaryTest(unittest.TestCase):
    def test_runtime_python_code_does_not_default_to_external_dirs(self) -> None:
        offenders: list[str] = []
        for top_level in ("app", "benchmark"):
            for path in sorted((ROOT / top_level).rglob("*.py")):
                if "/data/" in path.as_posix():
                    continue
                text = path.read_text()
                for pattern in BANNED_PATTERNS:
                    if pattern.search(text):
                        relative_path = str(path.relative_to(ROOT))
                        if relative_path not in ALLOWED_EXTERNAL_REFERENCE_PATHS:
                            offenders.append(relative_path)
                        break
        self.assertEqual(offenders, [], f"Found runtime code referencing External/external paths: {offenders}")


if __name__ == "__main__":
    unittest.main()
