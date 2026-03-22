from __future__ import annotations

import unittest

from app.codegen.catalog import seed_strategy_experiences
from app.configs.codegen import DEFAULT_SESSION_ID, DEFAULT_SPEEDUP_OBJECTIVE_SPEC, DISCRETE_DEMO_J_SPEC, J_FORMULA, TRAINER_J_SPEC


class CodegenDefaultsTest(unittest.TestCase):
    def test_seed_strategy_experiences_returns_copy(self) -> None:
        experiences = seed_strategy_experiences()
        self.assertEqual(len(experiences), 2)
        experiences[0]["experience_id"] = "mutated"
        self.assertNotEqual(seed_strategy_experiences()[0]["experience_id"], "mutated")

    def test_shared_scoring_specs_stay_aligned(self) -> None:
        self.assertEqual(TRAINER_J_SPEC["formula"], J_FORMULA)
        self.assertEqual(DISCRETE_DEMO_J_SPEC["formula"], J_FORMULA)
        self.assertEqual(DEFAULT_SESSION_ID, "session-current")

    def test_default_objective_spec_exposes_speedup_template(self) -> None:
        self.assertEqual(DEFAULT_SPEEDUP_OBJECTIVE_SPEC["display_name"], "Speedup vs baseline")
        self.assertEqual(
            DEFAULT_SPEEDUP_OBJECTIVE_SPEC["formula"],
            "speedup_vs_baseline = baseline_ms / candidate_ms",
        )


if __name__ == "__main__":
    unittest.main()
