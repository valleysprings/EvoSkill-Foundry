from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BANNED_PATTERNS = (
    re.compile(r'["\']external/'),
    re.compile(r'["\']External/'),
    re.compile(r'/\s*["\']external["\']'),
    re.compile(r'/\s*["\']External["\']'),
)


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
                        offenders.append(str(path.relative_to(ROOT)))
                        break
        self.assertEqual(offenders, [], f"Found runtime code referencing External/external paths: {offenders}")


if __name__ == "__main__":
    unittest.main()
