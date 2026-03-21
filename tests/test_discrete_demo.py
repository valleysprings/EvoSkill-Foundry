from __future__ import annotations

import unittest

from app.entries.discrete_demo import generate_discrete_payload


class DiscreteDemoPayloadTest(unittest.TestCase):
    def test_tsp_improves_over_baseline(self) -> None:
        payload = generate_discrete_payload(task_id="euclidean-tsp")
        run = payload["runs"][0]
        self.assertEqual(run["task"]["id"], "euclidean-tsp")
        self.assertLess(run["winner"]["metrics"]["objective"], run["baseline"]["metrics"]["objective"])
        self.assertGreater(run["delta_J"], 0.0)
        self.assertGreater(run["memory_after_count"], run["memory_before_count"])

    def test_max_cut_improves_over_baseline(self) -> None:
        payload = generate_discrete_payload(task_id="weighted-max-cut")
        run = payload["runs"][0]
        self.assertEqual(run["task"]["id"], "weighted-max-cut")
        self.assertGreater(run["winner"]["metrics"]["objective"], run["baseline"]["metrics"]["objective"])
        self.assertGreater(run["delta_J"], 0.0)
        self.assertTrue(payload["memory_markdown"].startswith("# Discrete Autoresearch Memory"))


if __name__ == "__main__":
    unittest.main()
