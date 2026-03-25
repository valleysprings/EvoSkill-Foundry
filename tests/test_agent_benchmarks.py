from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from app.codegen.agent_benchmarks import (
    _coerce_tau_results_payload,
    _legacy_tau_cli_is_likely,
    _load_tau_results,
)
from app.codegen.external import strip_socks_proxy_env


class AgentBenchmarksTest(unittest.TestCase):
    def test_strip_socks_proxy_env_removes_only_socks_entries(self) -> None:
        cleaned = strip_socks_proxy_env(
            {
                "ALL_PROXY": "socks5://127.0.0.1:7897",
                "HTTPS_PROXY": "http://proxy.example:8080",
                "NO_PROXY": "localhost",
            }
        )
        self.assertNotIn("ALL_PROXY", cleaned)
        self.assertEqual(cleaned["HTTPS_PROXY"], "http://proxy.example:8080")
        self.assertEqual(cleaned["NO_PROXY"], "localhost")

    def test_coerce_tau_results_accepts_top_level_list(self) -> None:
        rows = _coerce_tau_results_payload([{"task_id": "1", "reward": 1.0}])
        self.assertEqual(rows, [{"task_id": "1", "reward": 1.0}])

    def test_coerce_tau_results_accepts_named_result_list(self) -> None:
        rows = _coerce_tau_results_payload({"results": [{"task_id": "2", "success": True}]})
        self.assertEqual(rows, [{"task_id": "2", "success": True}])

    def test_load_tau_results_falls_back_to_repo_simulation_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            repo_dir = root / "repo"
            repo_dir.mkdir()
            simulation_dir = repo_dir / "data" / "tau2" / "simulations"
            simulation_dir.mkdir(parents=True)
            result_file = simulation_dir / "tau-retail-session.json"
            result_file.write_text(json.dumps([{"task_id": "3", "reward": 0.0}]))

            rows, resolved_path = _load_tau_results(root / "workspace" / "tau-retail-session.json", repo_dir)

            self.assertEqual(rows, [{"task_id": "3", "reward": 0.0}])
            self.assertEqual(resolved_path, result_file)

    def test_legacy_tau_cli_detection_requires_legacy_repo_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            repo_dir = Path(tmp_dir)
            (repo_dir / "run.py").write_text("print('legacy tau')")

            self.assertTrue(
                _legacy_tau_cli_is_likely(
                    repo_dir,
                    RuntimeError("Command failed: uv run tau2 run :: Failed to spawn: `tau2`"),
                )
            )
            self.assertFalse(
                _legacy_tau_cli_is_likely(
                    repo_dir,
                    RuntimeError("Command failed: uv run tau2 run :: missing API key"),
                )
            )


if __name__ == "__main__":
    unittest.main()
