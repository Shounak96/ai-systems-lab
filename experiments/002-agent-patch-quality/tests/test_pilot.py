import difflib
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pilot


class PilotTests(unittest.TestCase):
    def test_five_unique_categories(self):
        self.assertEqual(len(pilot.SCENARIOS), 5)
        self.assertEqual(len({item.category for item in pilot.SCENARIOS}), 5)

    def test_hidden_tests_are_not_in_prompts(self):
        for scenario in pilot.SCENARIOS:
            prompt = pilot.prompt_for(scenario)
            self.assertNotIn("HiddenTests", prompt)
            self.assertNotIn(scenario.hidden_tests, prompt)

    def test_root_cause_scoring_is_deterministic(self):
        self.assertTrue(pilot.root_cause_score("A shallow copy shares the list", ("shallow", "list")))
        self.assertFalse(pilot.root_cause_score("copy issue", ("shallow", "list")))

    def test_patch_scope_rejects_other_files(self):
        with self.assertRaises(ValueError):
            pilot.validate_patch("--- a/other.py\n+++ b/other.py\n@@ -1 +1 @@\n-x\n+y\n")

    def test_each_baseline_fails_its_public_test(self):
        for scenario in pilot.SCENARIOS:
            with self.subTest(scenario=scenario.id):
                namespace = {}
                exec(compile(scenario.source, "harness.py", "exec"), namespace)
                tests_ns = {"__name__": f"baseline_{scenario.id}"}
                # Imports are replaced so the public test uses this baseline namespace.
                body = scenario.public_tests.replace("from harness import ", "# from harness import ")
                for name, value in namespace.items():
                    if not name.startswith("__"):
                        tests_ns[name] = value
                exec(compile(body, "test_public.py", "exec"), tests_ns)
                suite = unittest.defaultTestLoader.loadTestsFromTestCase(tests_ns["PublicTests"])
                result = unittest.TestResult(); suite.run(result)
                self.assertFalse(result.wasSuccessful())

    def test_evaluator_accepts_a_reference_fix(self):
        scenario = next(item for item in pilot.SCENARIOS if item.id == "state_management")
        fixed = '''def add_message(state: dict, message: str) -> dict:
    """Return state updated with one history message."""
    updated = state.copy()
    updated["history"] = list(state.get("history", [])) + [message]
    return updated
'''
        patch = "".join(difflib.unified_diff(
            scenario.source.splitlines(True), fixed.splitlines(True),
            fromfile="a/harness.py", tofile="b/harness.py",
        ))
        applicable, public_fixed, regression_free, output = pilot.evaluate_patch(scenario, patch)
        self.assertTrue(applicable, output)
        self.assertTrue(public_fixed, output)
        self.assertTrue(regression_free, output)


if __name__ == "__main__":
    unittest.main()
