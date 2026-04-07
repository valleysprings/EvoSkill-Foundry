from __future__ import annotations

import unittest

from app.bench.personalization_references import (
    list_personalization_mirror_repos,
    load_personalization_reference_benchmarks,
)


class PersonalizationReferenceCatalogTest(unittest.TestCase):
    def test_reference_catalog_matches_official_protocol_reset(self) -> None:
        references = load_personalization_reference_benchmarks()
        self.assertEqual(len(references), 8)

        reference_ids = {entry["id"] for entry in references}
        self.assertEqual(
            reference_ids,
            {
                "incharacter",
                "characterbench",
                "socialbench",
                "timechara",
                "personafeedback",
                "personamem",
                "alpsbench",
                "alpbench",
            },
        )

        self.assertNotIn("roleeval", reference_ids)
        self.assertNotIn("ditto", reference_ids)
        self.assertNotIn("incharacter-acg", reference_ids)
        self.assertNotIn("characterbench-acg", reference_ids)
        self.assertNotIn("rmtbench", reference_ids)
        self.assertNotIn("charactereval", reference_ids)
        self.assertNotIn("coser", reference_ids)

        modes = {entry["interaction_mode"] for entry in references}
        self.assertEqual(modes, {"single_turn"})

        categories = {entry["benchmark_category"] for entry in references}
        self.assertEqual(categories, {"explicit_character_persona", "user_persona_personalization"})

        primary_categories = {entry["primary_category"] for entry in references}
        self.assertEqual(
            primary_categories,
            {
                "character_portrayal",
                "consistency_robustness",
                "user_personalization",
            },
        )

        running = {entry["id"] for entry in references if entry["implementation_status"] == "running"}
        planned = {entry["id"] for entry in references if entry["implementation_status"] in {"phase1", "phase2"}}
        blocked = {entry["id"] for entry in references if entry["implementation_status"] == "blocked"}
        local_task_ids = {entry["id"] for entry in references if entry["status"] == "local_task"}

        self.assertEqual(running, local_task_ids)
        self.assertEqual(
            running,
            {
                "incharacter",
                "characterbench",
                "socialbench",
                "timechara",
                "personafeedback",
                "personamem",
                "alpsbench",
                "alpbench",
            },
        )
        self.assertEqual(
            planned,
            set(),
        )
        self.assertEqual(blocked, set())

        self.assertEqual(
            {entry["id"] for entry in references if entry["status"] == "planned_task"},
            set(),
        )
        self.assertEqual(
            {entry["id"] for entry in references if entry["status"] == "external_reference"},
            set(),
        )

        self.assertEqual(
            {entry["id"] for entry in references if entry["metric_fidelity"] == "official"},
            {
                "incharacter",
                "characterbench",
                "socialbench",
                "timechara",
                "personafeedback",
                "personamem",
                "alpsbench",
                "alpbench",
            },
        )
        self.assertTrue(all(entry["metric_fidelity"] == "reference_only" for entry in references if entry["id"] not in running))
        self.assertEqual(
            {entry["id"] for entry in references if entry["interaction_mode"] == "multi_turn" and entry["implementation_status"] == "running"},
            set(),
        )

        for entry in references:
            self.assertTrue(entry["official_dimensions"])
            self.assertTrue(entry["protocol_summary"])
            self.assertTrue(entry["implementation_note"])
            self.assertTrue(entry["required_runtime_roles"])

        incharacter = next(entry for entry in references if entry["id"] == "incharacter")
        self.assertEqual(incharacter["status"], "local_task")
        self.assertEqual(incharacter["task_id"], "incharacter")
        self.assertEqual(incharacter["implementation_status"], "running")
        self.assertEqual(incharacter["task_shape"], "dialogue_judgement")
        self.assertTrue(incharacter["supports_eval_model"])
        self.assertTrue(incharacter["requires_eval_model"])
        self.assertIn("judge_model", incharacter["required_runtime_roles"])
        self.assertEqual(incharacter["metric_fidelity"], "official")

        characterbench = next(entry for entry in references if entry["id"] == "characterbench")
        self.assertEqual(characterbench["status"], "local_task")
        self.assertEqual(characterbench["task_id"], "characterbench")
        self.assertEqual(characterbench["official_metric_name"], "CharacterJudge score")
        self.assertTrue(characterbench["supports_eval_model"])
        self.assertTrue(characterbench["requires_eval_model"])
        self.assertIn("judge_model", characterbench["required_runtime_roles"])
        self.assertEqual(characterbench["metric_fidelity"], "official")

        socialbench = next(entry for entry in references if entry["id"] == "socialbench")
        self.assertEqual(socialbench["status"], "local_task")
        self.assertEqual(socialbench["task_id"], "socialbench")
        self.assertEqual(socialbench["implementation_status"], "running")
        self.assertEqual(socialbench["metric_fidelity"], "official")

        timechara = next(entry for entry in references if entry["id"] == "timechara")
        self.assertEqual(timechara["official_metric_backend"], "llm_judge")
        self.assertEqual(timechara["status"], "local_task")
        self.assertEqual(timechara["task_id"], "timechara")
        self.assertEqual(timechara["implementation_status"], "running")
        self.assertTrue(timechara["requires_eval_model"])
        self.assertEqual(timechara["metric_fidelity"], "official")

        personafeedback = next(entry for entry in references if entry["id"] == "personafeedback")
        self.assertEqual(personafeedback["status"], "local_task")
        self.assertEqual(personafeedback["task_id"], "personafeedback")
        self.assertEqual(personafeedback["metric_fidelity"], "official")

        personamem = next(entry for entry in references if entry["id"] == "personamem")
        self.assertEqual(personamem["task_id"], "personamem-32k")
        self.assertEqual(personamem["primary_category"], "user_personalization")

        alpbench = next(entry for entry in references if entry["id"] == "alpbench")
        self.assertEqual(alpbench["task_shape"], "classification")
        self.assertEqual(alpbench["scoring_mode"], "label_match")

    def test_mirror_repo_list_is_deduplicated_after_cleanup(self) -> None:
        repos = dict(list_personalization_mirror_repos())
        self.assertNotIn("roleeval", repos)
        self.assertNotIn("ditto", repos)
        self.assertIn("incharacter", repos)
        self.assertIn("characterbench", repos)
        self.assertIn("socialbench", repos)
        self.assertIn("timechara", repos)
        self.assertNotIn("charactereval", repos)
        self.assertNotIn("coser", repos)
        self.assertIn("personamem", repos)
        self.assertNotIn("personalens", repos)


if __name__ == "__main__":
    unittest.main()
