import json
import shutil
import sys
import unittest
import uuid
from dataclasses import asdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import v2_integration as v2


class V2IntegrationTests(unittest.TestCase):
    def make_run_dir(self) -> Path:
        directory = ROOT / ".tmp" / f"t-{uuid.uuid4().hex[:8]}"
        directory.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, directory, True)
        return directory / "run"

    def test_official_source_is_pinned_and_clean(self):
        self.assertTrue((v2.AUTOSADDLER_ROOT / ".git").is_dir())
        self.assertEqual(v2.main_check_pin_for_test(), v2.AUTOSADDLER_COMMIT)

    def test_baseline_and_reference_behavior(self):
        baseline_train, _, _ = v2.execute_case(v2.BASELINE_SOURCE, v2.TRAIN_CASE_ID)
        baseline_invalid, _, _ = v2.execute_case(v2.BASELINE_SOURCE, "dev-invalid-values")
        baseline_preserved, _, _ = v2.execute_case(v2.BASELINE_SOURCE, "dev-dict-preservation")
        self.assertFalse(baseline_train)
        self.assertFalse(baseline_invalid)
        self.assertTrue(baseline_preserved)
        for case_id in (v2.TRAIN_CASE_ID, *v2.DEV_CASE_IDS):
            passed, output, _ = v2.execute_case(v2.REFERENCE_SOURCE, case_id)
            self.assertTrue(passed, output)

    def test_failure_trace_removes_volatile_workspace_path(self):
        first_passed, first_output, _ = v2.execute_case(v2.BASELINE_SOURCE, v2.TRAIN_CASE_ID)
        second_passed, second_output, _ = v2.execute_case(v2.BASELINE_SOURCE, v2.TRAIN_CASE_ID)
        self.assertFalse(first_passed)
        self.assertFalse(second_passed)
        self.assertEqual(first_output, second_output)
        self.assertIn("<isolated-workspace>", first_output)
        self.assertNotIn(str(v2.TEMP_ROOT), first_output)

    def test_verifier_rejects_io(self):
        with self.assertRaises(ValueError):
            v2.validate_component_source({"harness.py": "import os\ndef normalize_arguments(arguments):\n return os.getenv('X')\n"})

    def test_hidden_development_expectations_are_not_in_diagnosis_prompt(self):
        run_dir = self.make_run_dir()
        store = v2.LocalRunStore(run_dir=run_dir, run_id="prompt-isolation")
        scenario = v2.build_scenario(run_dir, store)
        store.initialize(resolved_config={"test": True}, resolved_entities=scenario.resolved_entities)
        seed = scenario.harness_space.seed()
        evidence = store.write_json(
            "evidence/test/evidence.json",
            {
                "schema_version": "exp002-training-evidence/v1",
                "candidate_id": seed.candidate_id,
                "split": "train",
                "case_records": [{"case_id": v2.TRAIN_CASE_ID, "score": 0.0}],
            },
            kind="test-evidence",
        )
        spec = scenario.prompt_pack.session(
            "diagnose_patch",
            {"iteration": 0, "candidate_ids": [seed.candidate_id], "train_case_ids": [v2.TRAIN_CASE_ID], "evidence": asdict(evidence)},
        )
        serialized = json.dumps({"prompt": spec.task_prompt, "files": dict(spec.workspace_files)})
        for case_id in v2.DEV_CASE_IDS:
            self.assertNotIn(case_id, serialized)
        self.assertNotIn("[1, 2]", serialized)

    def test_reference_provider_runs_official_workflow(self):
        run_dir = self.make_run_dir()
        response = {
            "schema_version": "exp002-diagnosis-update/v1",
            "diagnosis": "JSON strings are passed directly to dict instead of being decoded and validated.",
            "expected_effect": "Accept object strings and dicts while rejecting non-object values.",
            "updates": {"harness.py": v2.REFERENCE_SOURCE},
        }
        provider = v2.HybridOllamaProvider("reference", diagnosis_override=lambda request: response)
        engine, store = v2.build_engine(run_dir, "reference-workflow", provider)
        result = engine.run()
        summary = v2.summarize_run("reference", store, provider)
        self.assertEqual(provider.diagnosis_calls, 1)
        self.assertTrue(summary["schema_valid"])
        self.assertTrue(summary["mutation_finalized"])
        self.assertTrue(summary["accepted_mutation"])
        self.assertEqual(summary["training_improvement"], 1.0)
        self.assertEqual(summary["development_improvement"], 0.5)
        self.assertEqual(summary["development_regressions"], [])
        self.assertEqual(result.development_score, 1.0)


if __name__ == "__main__":
    unittest.main()
