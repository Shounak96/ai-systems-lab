import json
import shutil
import sys
import unittest
import uuid
from dataclasses import asdict
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import v4_evaluation as v4


class V4EvaluationTests(unittest.TestCase):
    def make_run_dir(self) -> Path:
        directory = ROOT / ".tmp" / "v4tests" / f"t-{uuid.uuid4().hex[:8]}"
        directory.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, directory, True)
        return directory / "run"

    def test_fifteen_independent_scenarios_and_splits(self):
        self.assertEqual(len(v4.SCENARIOS), 15)
        self.assertEqual(len({item.key for item in v4.SCENARIOS}), 15)
        train = {item.train_case for item in v4.SCENARIOS}
        development = {case for item in v4.SCENARIOS for case in item.development_cases}
        heldout = {case for cases in v4.TEST_CASES.values() for case in cases}
        self.assertEqual(len(train), 15)
        self.assertEqual(len(development), 30)
        self.assertEqual(len(heldout), 30)
        self.assertFalse(train & development or train & heldout or development & heldout)

    def test_baselines_fail_training_and_references_solve_all_splits(self):
        for definition in v4.SCENARIOS:
            passed, _, _ = v4.v3.execute_case(definition, definition.baseline, definition.train_case)
            self.assertFalse(passed, definition.key)
            all_cases = (definition.train_case, *definition.development_cases, *v4.TEST_CASES[definition.key])
            for case_id in all_cases:
                passed, output, _ = v4.v3.execute_case(definition, definition.reference, case_id)
                self.assertTrue(passed, f"{definition.key}/{case_id}: {output}")

    def test_sanitized_training_evidence_is_stable(self):
        for definition in v4.SCENARIOS:
            first = v4.v3.execute_case(definition, definition.baseline, definition.train_case)[1]
            second = v4.v3.execute_case(definition, definition.baseline, definition.train_case)[1]
            self.assertEqual(first.encode(), second.encode(), definition.key)
            self.assertIn("<isolated-workspace>", first, definition.key)
            self.assertNotIn(str(v4.v3.TEMP_ROOT), first, definition.key)

    def test_heldout_cases_are_absent_from_engine_scenario_and_prompt(self):
        for definition in v4.SCENARIOS:
            run_dir = self.make_run_dir()
            store = v4.v2.ProjectionRetryStore(run_dir=run_dir, run_id=f"leak-{definition.short}")
            components = v4.v3.build_scenario(definition, run_dir, store)
            store.initialize(resolved_config={"test": True}, resolved_entities=components.resolved_entities)
            engine_case_ids = {case.case_id for case in (*components.train_cases, *components.development_cases)}
            self.assertFalse(engine_case_ids & set(v4.TEST_CASES[definition.key]))
            seed = components.harness_space.seed()
            evidence = store.write_json(
                "evidence/test/evidence.json",
                {"schema_version": "exp002-training-evidence/v1", "candidate_id": seed.candidate_id,
                 "split": "train", "case_records": [{"case_id": definition.train_case, "score": 0.0}]},
                kind="test-evidence",
            )
            spec = components.prompt_pack.session(
                "diagnose_patch",
                {"iteration": 0, "candidate_ids": [seed.candidate_id], "train_case_ids": [definition.train_case], "evidence": asdict(evidence)},
            )
            serialized = v4.v2.canonical_json(
                {"prompt": spec.task_prompt, "files": dict(spec.workspace_files), "resolved": dict(components.resolved_entities)}
            )
            for case_id in v4.TEST_CASES[definition.key]:
                self.assertNotIn(case_id, serialized, definition.key)

    def test_normalized_provider_payloads_match_for_every_scenario(self):
        for definition in v4.SCENARIOS:
            run_dir = self.make_run_dir()
            store = v4.v2.ProjectionRetryStore(run_dir=run_dir, run_id=f"payload-{definition.short}")
            components = v4.v3.build_scenario(definition, run_dir, store)
            store.initialize(resolved_config={"test": True}, resolved_entities=components.resolved_entities)
            seed = components.harness_space.seed()
            evidence = store.write_json(
                "evidence/test/evidence.json",
                {"schema_version": "exp002-training-evidence/v1", "candidate_id": seed.candidate_id,
                 "split": "train", "case_records": [{"case_id": definition.train_case, "score": 0.0}]},
                kind="test-evidence",
            )
            spec = components.prompt_pack.session(
                "diagnose_patch",
                {"iteration": 0, "candidate_ids": [seed.candidate_id], "train_case_ids": [definition.train_case], "evidence": asdict(evidence)},
            )
            workspace = run_dir / "payload"; workspace.mkdir()
            (workspace / "candidate.json").write_text(json.dumps({"harness.py": definition.baseline}), encoding="utf-8")
            request = v4.v2.SessionRequest(
                session_id="test", operation_id="test:1", spec=spec, workspace=workspace,
                timeout_seconds=30, trace_dir=run_dir / "trace",
            )
            first = v4.v2.HybridOllamaProvider(v4.MODELS[0]).build_ollama_payload(request)
            second = v4.v2.HybridOllamaProvider(v4.MODELS[1]).build_ollama_payload(request)
            first["model"] = second["model"] = "MODEL"
            self.assertEqual(v4.v2.canonical_json(first).encode(), v4.v2.canonical_json(second).encode())

    def test_projection_retry_is_bounded_and_storage_only(self):
        store = v4.v2.ProjectionRetryStore(run_dir=self.make_run_dir(), run_id="retry-store")
        with mock.patch.object(
            v4.v2.LocalRunStore,
            "refresh_projections",
            side_effect=[PermissionError("locked"), PermissionError("locked"), None],
        ) as refresh:
            store.refresh_projections()
        self.assertEqual(refresh.call_count, 3)
        self.assertEqual(store.projection_retry_limit, 8)

    def test_reference_fixes_pass_official_workflow_and_post_selection_test(self):
        for definition in v4.SCENARIOS:
            run_dir = self.make_run_dir()
            response = {
                "schema_version": "exp002-diagnosis-update/v1",
                "diagnosis": "Known-good fixture; never sent to an optimizer model.",
                "expected_effect": "Resolve all frozen cases.",
                "updates": {"harness.py": definition.reference},
            }
            provider = v4.v2.HybridOllamaProvider(
                "reference", diagnosis_override=lambda request, value=response: value
            )
            engine, store, components = v4.build_engine(
                definition, run_dir, f"reference-{definition.short}", provider
            )
            result = engine.run()
            summary = v4.v3.summarize_run(definition, "reference", store, provider)
            heldout = v4.evaluate_heldout(store, components, result, definition)
            self.assertTrue(summary["schema_valid"], definition.key)
            self.assertTrue(summary["mutation_finalized"], definition.key)
            self.assertTrue(summary["accepted_mutation"], definition.key)
            self.assertEqual(summary["training_improvement"], 1.0, definition.key)
            self.assertEqual(summary["development_score_after"], 1.0, definition.key)
            self.assertEqual(heldout["heldout_score_after"], 1.0, definition.key)
            self.assertEqual(heldout["heldout_regressions"], [], definition.key)

    def test_verifier_rejects_filesystem_access(self):
        definition = v4.SCENARIO_BY_KEY["error_handling"]
        with self.assertRaises(ValueError):
            v4.v3.validate_component_source(
                definition,
                {"harness.py": "import os\ndef safe_error_message(error):\n return os.getenv('X')\n"},
            )


if __name__ == "__main__":
    unittest.main()
