import json
import shutil
import sys
import unittest
import uuid
from dataclasses import asdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import v3_integration as v3


class V3IntegrationTests(unittest.TestCase):
    def make_run_dir(self) -> Path:
        directory = ROOT / ".tmp" / "v3tests" / f"t-{uuid.uuid4().hex[:8]}"
        directory.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, directory, True)
        return directory / "run"

    def test_five_frozen_categories(self):
        self.assertEqual(
            [item.key for item in v3.SCENARIOS],
            ["structured_output", "tool_arguments", "retries", "state_management", "error_handling"],
        )
        self.assertEqual(len({item.train_case for item in v3.SCENARIOS}), 5)
        self.assertEqual(len({case for item in v3.SCENARIOS for case in item.development_cases}), 10)

    def test_sanitized_training_traces_are_byte_identical(self):
        for definition in v3.SCENARIOS:
            first_passed, first_output, _ = v3.execute_case(
                definition, definition.baseline, definition.train_case
            )
            second_passed, second_output, _ = v3.execute_case(
                definition, definition.baseline, definition.train_case
            )
            self.assertFalse(first_passed, definition.key)
            self.assertFalse(second_passed, definition.key)
            self.assertEqual(first_output.encode("utf-8"), second_output.encode("utf-8"), definition.key)
            self.assertIn("<isolated-workspace>", first_output, definition.key)
            self.assertNotIn(str(v3.TEMP_ROOT), first_output, definition.key)

    def test_provider_payloads_are_byte_identical_except_model_identity(self):
        definition = v3.SCENARIO_BY_KEY["tool_arguments"]
        run_dir = self.make_run_dir()
        store = v3.v2.LocalRunStore(run_dir=run_dir, run_id="payload-identity")
        scenario = v3.build_scenario(definition, run_dir, store)
        store.initialize(resolved_config={"test": True}, resolved_entities=scenario.resolved_entities)
        seed = scenario.harness_space.seed()
        evidence = store.write_json(
            "evidence/test/evidence.json",
            {
                "schema_version": "exp002-training-evidence/v1",
                "candidate_id": seed.candidate_id,
                "split": "train",
                "case_records": [{"case_id": definition.train_case, "score": 0.0}],
            },
            kind="test-evidence",
        )
        spec = scenario.prompt_pack.session(
            "diagnose_patch",
            {
                "iteration": 0,
                "candidate_ids": [seed.candidate_id],
                "train_case_ids": [definition.train_case],
                "evidence": asdict(evidence),
            },
        )
        workspace = run_dir / "payload-workspace"
        workspace.mkdir()
        (workspace / "candidate.json").write_text(
            json.dumps({"harness.py": definition.baseline}), encoding="utf-8"
        )
        request = v3.v2.SessionRequest(
            session_id="payload-test",
            operation_id="payload-test:1",
            spec=spec,
            workspace=workspace,
            timeout_seconds=30,
            trace_dir=run_dir / "trace",
        )
        first = v3.v2.HybridOllamaProvider("qwen3:1.7b").build_ollama_payload(request)
        second = v3.v2.HybridOllamaProvider("qwen3:4b").build_ollama_payload(request)
        first["model"] = "MODEL"
        second["model"] = "MODEL"
        self.assertEqual(v3.v2.canonical_json(first).encode(), v3.v2.canonical_json(second).encode())

    def test_known_good_mutations_pass_all_cases(self):
        for definition in v3.SCENARIOS:
            baseline_passed, _, _ = v3.execute_case(
                definition, definition.baseline, definition.train_case
            )
            self.assertFalse(baseline_passed, definition.key)
            for case_id in (definition.train_case, *definition.development_cases):
                passed, output, _ = v3.execute_case(definition, definition.reference, case_id)
                self.assertTrue(passed, f"{definition.key}/{case_id}: {output}")

    def test_verifier_rejects_unregistered_io(self):
        definition = v3.SCENARIO_BY_KEY["error_handling"]
        with self.assertRaises(ValueError):
            v3.validate_component_source(
                definition,
                {"harness.py": "import os\ndef safe_error_message(error):\n return os.getenv('X')\n"},
            )

    def test_development_expectations_do_not_enter_prompt(self):
        for definition in v3.SCENARIOS:
            run_dir = self.make_run_dir()
            store = v3.v2.LocalRunStore(run_dir=run_dir, run_id=f"prompt-{definition.short}")
            scenario = v3.build_scenario(definition, run_dir, store)
            store.initialize(resolved_config={"test": True}, resolved_entities=scenario.resolved_entities)
            seed = scenario.harness_space.seed()
            evidence = store.write_json(
                "evidence/test/evidence.json",
                {
                    "schema_version": "exp002-training-evidence/v1",
                    "candidate_id": seed.candidate_id,
                    "split": "train",
                    "case_records": [{"case_id": definition.train_case, "score": 0.0}],
                },
                kind="test-evidence",
            )
            spec = scenario.prompt_pack.session(
                "diagnose_patch",
                {
                    "iteration": 0,
                    "candidate_ids": [seed.candidate_id],
                    "train_case_ids": [definition.train_case],
                    "evidence": asdict(evidence),
                },
            )
            serialized = json.dumps(
                {"system": spec.system_context, "prompt": spec.task_prompt, "files": dict(spec.workspace_files)}
            )
            for case_id in definition.development_cases:
                self.assertNotIn(case_id, serialized, definition.key)
            case_block = v3.CASE_BLOCKS[definition.key]
            for sentinel in ("DEV_PASSWORD", "worker-7", "units", "meta"):
                if sentinel in case_block:
                    self.assertNotIn(sentinel, serialized, definition.key)

    def test_reference_mutations_use_official_workflow(self):
        for definition in v3.SCENARIOS:
            run_dir = self.make_run_dir()
            response = {
                "schema_version": "exp002-diagnosis-update/v1",
                "diagnosis": "Known-good integration fixture, never model-facing.",
                "expected_effect": "Resolve training and development cases.",
                "updates": {"harness.py": definition.reference},
            }
            provider = v3.v2.HybridOllamaProvider(
                "reference", diagnosis_override=lambda request, value=response: value
            )
            engine, store = v3.build_engine(
                definition, run_dir, f"reference-{definition.short}", provider
            )
            result = engine.run()
            summary = v3.v2.summarize_run("reference", store, provider)
            self.assertEqual(provider.diagnosis_calls, 1, definition.key)
            self.assertTrue(summary["schema_valid"], definition.key)
            self.assertTrue(summary["mutation_finalized"], definition.key)
            self.assertTrue(summary["accepted_mutation"], definition.key)
            self.assertEqual(summary["training_improvement"], 1.0, definition.key)
            self.assertTrue(summary["development_evaluated"], definition.key)
            self.assertEqual(summary["development_score_after"], 1.0, definition.key)
            self.assertEqual(summary["development_regressions"], [], definition.key)
            self.assertEqual(result.development_score, 1.0, definition.key)


if __name__ == "__main__":
    unittest.main()
