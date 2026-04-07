from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.bench.benchmark_adapter_support import run_benchmark_adapter_task
from tests.helpers import make_runtime


class BenchmarkAdapterMemoryTest(unittest.TestCase):
    def test_run_benchmark_adapter_task_forwards_memory_root_to_supported_verifier(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            editable_path = root / "editable.py"
            verifier_path = root / "verifier.py"
            editable_path.write_text("from __future__ import annotations\n\nVALUE = 'ok'\n")
            verifier_path.write_text(
                "from __future__ import annotations\n\n"
                "def run_benchmark_adapter_task(*, task, candidate_path, source_code, proposal_runtime, workspace_root, memory_root, session_id, max_items, max_episodes, progress_callback, pace_ms):\n"
                "    del candidate_path, source_code, proposal_runtime, workspace_root, session_id, max_items, max_episodes, progress_callback, pace_ms\n"
                "    return {'memory_root': str(memory_root), 'task_id': task['id']}\n"
            )
            task = {
                "id": "memory-forwarding-fixture",
                "editable_path": str(editable_path),
                "verifier_path": str(verifier_path),
            }
            memory_root = root / "item-memory"
            result = run_benchmark_adapter_task(
                task,
                proposal_runtime=make_runtime([]),
                workspace_root=root / "workspace",
                memory_root=memory_root,
                session_id="session-current",
                max_items=1,
                max_episodes=None,
            )
            self.assertEqual(result, {"memory_root": str(memory_root), "task_id": "memory-forwarding-fixture"})


if __name__ == "__main__":
    unittest.main()
