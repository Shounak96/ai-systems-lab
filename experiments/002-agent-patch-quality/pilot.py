from __future__ import annotations

import argparse
import ast
import csv
import ctypes
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
RESULT_COLUMNS = [
    "model", "scenario", "category", "status", "root_cause_accurate",
    "patch_applicable", "failing_test_resolved", "regression_free",
    "prompt_tokens", "generated_tokens", "latency_seconds", "error",
]
MODEL_PREFERENCES = ("qwen3:1.7b", "qwen3:4b")
SAFE_IMPORTS = {"collections", "copy", "dataclasses", "json", "re", "typing"}
FORBIDDEN_CALLS = {"open", "exec", "eval", "compile", "__import__"}


@dataclass(frozen=True)
class Scenario:
    id: str
    category: str
    task: str
    trace: str
    source: str
    public_tests: str
    hidden_tests: str
    root_cause_terms: tuple[str, ...]


SCENARIOS = (
    Scenario(
        "structured_output", "structured-output validation",
        "Make parse_response accept a JSON object either directly or inside one Markdown JSON fence. Reject non-object JSON and malformed or extra prose with ValueError.",
        "parse_response('```json\\n{\"answer\": 7}\\n```') -> JSONDecodeError: Expecting value: line 1 column 1",
        '''import json\n\ndef parse_response(text: str) -> dict:\n    """Parse an agent response into one JSON object."""\n    value = json.loads(text)\n    if not isinstance(value, dict):\n        raise ValueError("response must be a JSON object")\n    return value\n''',
        '''import unittest\nfrom harness import parse_response\n\nclass PublicTests(unittest.TestCase):\n    def test_fenced_object(self):\n        self.assertEqual(parse_response('```json\\n{"answer": 7}\\n```'), {"answer": 7})\n''',
        '''import unittest\nfrom harness import parse_response\n\nclass HiddenTests(unittest.TestCase):\n    def test_plain_object(self):\n        self.assertEqual(parse_response('{"ok": true}'), {"ok": True})\n    def test_array_rejected(self):\n        with self.assertRaises(ValueError): parse_response('[1, 2]')\n    def test_extra_prose_rejected(self):\n        with self.assertRaises(ValueError): parse_response('note\\n{"ok": true}')\n    def test_unclosed_fence_rejected(self):\n        with self.assertRaises(ValueError): parse_response('```json\\n{"ok": true}')\n''',
        ("fence", "json"),
    ),
    Scenario(
        "tool_arguments", "tool argument handling",
        "Normalize tool arguments. Accept a dict or a JSON string containing an object; return a new dict. Raise ValueError for malformed JSON or any non-object value.",
        "normalize_arguments('{\"city\": \"Chicago\"}') -> ValueError: dictionary update sequence element #0 has length 1; 2 is required",
        '''def normalize_arguments(arguments) -> dict:\n    """Return normalized tool-call arguments."""\n    return dict(arguments)\n''',
        '''import unittest\nfrom harness import normalize_arguments\n\nclass PublicTests(unittest.TestCase):\n    def test_json_string(self):\n        self.assertEqual(normalize_arguments('{"city": "Chicago"}'), {"city": "Chicago"})\n''',
        '''import unittest\nfrom harness import normalize_arguments\n\nclass HiddenTests(unittest.TestCase):\n    def test_dict_is_copied(self):\n        source = {"x": 1}; result = normalize_arguments(source)\n        self.assertEqual(result, source); self.assertIsNot(result, source)\n    def test_json_array_rejected(self):\n        with self.assertRaises(ValueError): normalize_arguments('[1, 2]')\n    def test_malformed_rejected(self):\n        with self.assertRaises(ValueError): normalize_arguments('{bad')\n    def test_none_rejected(self):\n        with self.assertRaises(ValueError): normalize_arguments(None)\n''',
        ("string", "dict"),
    ),
    Scenario(
        "retries", "retries",
        "Fix retry_call so max_attempts is the total number of calls, retrying only TimeoutError. Require max_attempts >= 1 and re-raise the last TimeoutError after exhaustion.",
        "retry_call(flaky, max_attempts=3) calls flaky only 2 times, then returns None",
        '''def retry_call(operation, max_attempts: int = 3):\n    """Call operation, retrying transient timeouts."""\n    for _ in range(1, max_attempts):\n        try:\n            return operation()\n        except Exception:\n            pass\n    return None\n''',
        '''import unittest\nfrom harness import retry_call\n\nclass PublicTests(unittest.TestCase):\n    def test_third_attempt_succeeds(self):\n        calls = []\n        def operation():\n            calls.append(1)\n            if len(calls) < 3: raise TimeoutError("transient")\n            return "ok"\n        self.assertEqual(retry_call(operation, 3), "ok")\n        self.assertEqual(len(calls), 3)\n''',
        '''import unittest\nfrom harness import retry_call\n\nclass HiddenTests(unittest.TestCase):\n    def test_non_timeout_not_retried(self):\n        calls=[]\n        def operation(): calls.append(1); raise ValueError("bad input")\n        with self.assertRaises(ValueError): retry_call(operation, 3)\n        self.assertEqual(len(calls), 1)\n    def test_last_timeout_propagates(self):\n        calls=[]\n        def operation(): calls.append(1); raise TimeoutError("down")\n        with self.assertRaises(TimeoutError): retry_call(operation, 2)\n        self.assertEqual(len(calls), 2)\n    def test_invalid_attempt_count(self):\n        with self.assertRaises(ValueError): retry_call(lambda: 1, 0)\n''',
        ("off-by-one", "exception"),
    ),
    Scenario(
        "state_management", "state management",
        "Make add_message return a new state with the message appended, without mutating the input state or sharing its history list. Preserve all other keys.",
        "After add_message(state, 'new'), state['history'] unexpectedly changes from ['old'] to ['old', 'new']",
        '''def add_message(state: dict, message: str) -> dict:\n    """Return state updated with one history message."""\n    updated = state.copy()\n    updated.setdefault("history", []).append(message)\n    return updated\n''',
        '''import unittest\nfrom harness import add_message\n\nclass PublicTests(unittest.TestCase):\n    def test_input_not_mutated(self):\n        state = {"history": ["old"]}\n        result = add_message(state, "new")\n        self.assertEqual(state, {"history": ["old"]})\n        self.assertEqual(result["history"], ["old", "new"])\n''',
        '''import unittest\nfrom harness import add_message\n\nclass HiddenTests(unittest.TestCase):\n    def test_history_not_shared(self):\n        state={"history": []}; result=add_message(state, "x"); result["history"].append("y")\n        self.assertEqual(state["history"], [])\n    def test_missing_history(self):\n        state={"run_id": 4}; result=add_message(state, "x")\n        self.assertEqual(result, {"run_id": 4, "history": ["x"]}); self.assertEqual(state, {"run_id": 4})\n    def test_other_keys_preserved(self):\n        self.assertEqual(add_message({"history": [], "done": False}, "x")["done"], False)\n''',
        ("shallow", "list"),
    ),
    Scenario(
        "error_handling", "error handling",
        "Make safe_tool_call return {'ok': True, 'value': result} on success. For ordinary Exceptions return {'ok': False, 'error_type': class name, 'message': text}. Do not swallow BaseException subclasses such as KeyboardInterrupt.",
        "safe_tool_call(interrupting_tool) returns {'ok': False, 'error': ''} instead of propagating KeyboardInterrupt",
        '''def safe_tool_call(tool, *args, **kwargs) -> dict:\n    """Invoke a tool and represent ordinary failures as data."""\n    try:\n        return {"ok": True, "value": tool(*args, **kwargs)}\n    except BaseException as exc:\n        return {"ok": False, "error": str(exc)}\n''',
        '''import unittest\nfrom harness import safe_tool_call\n\nclass PublicTests(unittest.TestCase):\n    def test_keyboard_interrupt_propagates(self):\n        def tool(): raise KeyboardInterrupt()\n        with self.assertRaises(KeyboardInterrupt): safe_tool_call(tool)\n''',
        '''import unittest\nfrom harness import safe_tool_call\n\nclass HiddenTests(unittest.TestCase):\n    def test_success(self):\n        self.assertEqual(safe_tool_call(lambda x: x + 1, 2), {"ok": True, "value": 3})\n    def test_exception_shape(self):\n        def tool(): raise RuntimeError("offline")\n        self.assertEqual(safe_tool_call(tool), {"ok": False, "error_type": "RuntimeError", "message": "offline"})\n    def test_system_exit_propagates(self):\n        def tool(): raise SystemExit(2)\n        with self.assertRaises(SystemExit): safe_tool_call(tool)\n''',
        ("baseexception", "exception"),
    ),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def api(path: str, payload: dict[str, Any] | None = None, timeout: int = 10) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"http://127.0.0.1:11434{path}", data=data,
        headers={"Content-Type": "application/json"}, method="GET" if data is None else "POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def discover_models() -> tuple[list[str], str | None]:
    if shutil.which("ollama") is None:
        try:
            api("/api/tags", timeout=2)
        except Exception:
            return [], "Ollama executable not found on PATH and no server responded at 127.0.0.1:11434"
    try:
        tags = api("/api/tags", timeout=5)
    except Exception as exc:
        return [], f"Ollama server unavailable: {type(exc).__name__}: {exc}"
    names = {item["name"] for item in tags.get("models", [])}
    selected = []
    for preferred in MODEL_PREFERENCES:
        base = preferred.split(":", 1)[0]
        size = preferred.split(":", 1)[1]
        match = next((n for n in sorted(names) if n == preferred or (n.startswith(base + ":") and size in n)), None)
        if match:
            selected.append(match)
    if len(selected) < 2:
        return selected, f"Need two local Qwen3 sizes ({', '.join(MODEL_PREFERENCES)}); locally matched: {selected or 'none'}"
    return selected[:2], None


def model_metadata(model: str) -> dict[str, Any]:
    shown = api("/api/show", {"model": model}, timeout=30)
    details = shown.get("details", {})
    tags = api("/api/tags", timeout=10).get("models", [])
    tagged = next((item for item in tags if item.get("name") == model), {})
    return {
        "name": model,
        "family": details.get("family"),
        "parameter_size": details.get("parameter_size"),
        "quantization": details.get("quantization_level"),
        "digest": tagged.get("digest"),
    }


def prompt_for(scenario: Scenario) -> str:
    return f'''You diagnose and patch a small Python agent harness.\n\nTASK:\n{scenario.task}\n\nFAILURE TRACE:\n{scenario.trace}\n\nCURRENT harness.py:\n```python\n{scenario.source}```\n\nReturn exactly one JSON object with two string fields:\n- "root_cause": a concise diagnosis\n- "patch": a unified diff that modifies only harness.py, using paths a/harness.py and b/harness.py\n\nConstraints: preserve the public API, use only the Python standard library, make the smallest robust change, do not add tests or dependencies, do not access files/network/processes/environment, and output no Markdown fences or commentary outside the JSON. Hidden regression tests exist; their expectations are not provided.'''


def parse_model_output(raw: str) -> dict[str, str]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    value = json.loads(text)
    if set(value) != {"root_cause", "patch"} or not all(isinstance(value[k], str) for k in value):
        raise ValueError("response must contain exactly string fields root_cause and patch")
    return value


def root_cause_score(diagnosis: str, terms: tuple[str, ...]) -> bool:
    lowered = diagnosis.casefold()
    return all(term.casefold() in lowered for term in terms)


def validate_patch(patch: str) -> None:
    headers = re.findall(r"^(?:---|\+\+\+)\s+([^\t\n]+)", patch, re.MULTILINE)
    if not headers or any(path not in {"a/harness.py", "b/harness.py"} for path in headers):
        raise ValueError("patch must modify only harness.py with a/ and b/ paths")


def validate_source(path: Path) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [alias.name for alias in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            if any(name.split(".")[0] not in SAFE_IMPORTS for name in names):
                raise ValueError("patched source imports a module outside the evaluator allowlist")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALLS:
            raise ValueError(f"patched source calls forbidden builtin {node.func.id}")


RUNNER = '''import os\nimport socket\nimport sys\nimport unittest\n\nclass BlockedSocket:\n    def __init__(self, *args, **kwargs):\n        raise RuntimeError("network disabled in patch evaluator")\n\nsocket.socket = BlockedSocket\nsocket.create_connection = BlockedSocket\nsys.path.insert(0, os.path.dirname(__file__))\npattern = sys.argv[1] if len(sys.argv) > 1 else "test_*.py"\nsuite = unittest.defaultTestLoader.discover("tests", pattern=pattern)\nresult = unittest.TextTestRunner(verbosity=2).run(suite)\nsys.exit(0 if result.wasSuccessful() else 1)\n'''


def evaluate_patch(scenario: Scenario, patch: str) -> tuple[bool, bool, bool, str]:
    temp_root = ARTIFACTS / "tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    work = temp_root / f"exp002-{scenario.id}-{uuid.uuid4().hex}"
    work.mkdir()
    try:
        (work / "tests").mkdir()
        (work / "harness.py").write_text(scenario.source, encoding="utf-8")
        (work / "tests" / "test_public.py").write_text(scenario.public_tests, encoding="utf-8")
        (work / "tests" / "test_hidden.py").write_text(scenario.hidden_tests, encoding="utf-8")
        (work / "runner.py").write_text(RUNNER, encoding="utf-8")
        patch_path = work / "candidate.patch"
        patch_path.write_text(patch, encoding="utf-8")
        validate_patch(patch)
        checked = subprocess.run(
            ["git", "apply", "--check", "--whitespace=error", str(patch_path)], cwd=work,
            capture_output=True, text=True, timeout=10,
        )
        if checked.returncode:
            return False, False, False, checked.stderr.strip() or checked.stdout.strip()
        applied = subprocess.run(["git", "apply", str(patch_path)], cwd=work, capture_output=True, text=True, timeout=10)
        if applied.returncode:
            return False, False, False, applied.stderr.strip()
        validate_source(work / "harness.py")
        safe_temp = work / "tmp"
        safe_temp.mkdir()
        env = {
            "PATH": str(Path(sys.executable).parent), "PYTHONIOENCODING": "utf-8",
            "PYTHONDONTWRITEBYTECODE": "1", "TEMP": str(safe_temp), "TMP": str(safe_temp),
        }
        if os.name == "nt":
            env["SYSTEMROOT"] = os.environ.get("SYSTEMROOT", r"C:\Windows")
            env["WINDIR"] = os.environ.get("WINDIR", r"C:\Windows")
        public_tested = subprocess.run(
            [sys.executable, "-I", "runner.py", "test_public.py"], cwd=work, env=env,
            capture_output=True, text=True, timeout=20,
        )
        all_tested = subprocess.run(
            [sys.executable, "-I", "runner.py"], cwd=work, env=env,
            capture_output=True, text=True, timeout=20,
        )
        output = "PUBLIC TESTS:\n" + (public_tested.stdout + public_tested.stderr).strip()
        output += "\n\nFULL TEST SUITE:\n" + (all_tested.stdout + all_tested.stderr).strip()
        return True, public_tested.returncode == 0, all_tested.returncode == 0, output
    finally:
        # Resolve and verify scope before recursively removing the ephemeral copy.
        if work.resolve().parent == temp_root.resolve():
            shutil.rmtree(work, ignore_errors=True)


def hardware_info() -> dict[str, Any]:
    class MemoryStatus(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
            (name, ctypes.c_ulonglong) for name in (
                "total_phys", "avail_phys", "total_page", "avail_page",
                "total_virtual", "avail_virtual", "avail_extended",
            )
        ]
    memory = MemoryStatus(); memory.length = ctypes.sizeof(MemoryStatus)
    total = available = None
    if os.name == "nt" and ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
        total, available = round(memory.total_phys / 2**30, 2), round(memory.avail_phys / 2**30, 2)
    return {
        "platform": platform.platform(), "python": platform.python_version(),
        "python_executable": sys.executable, "cpu": platform.processor(),
        "logical_cpu_count": os.cpu_count(), "ram_gib": total,
        "available_ram_gib_at_start": available,
        "gpu": "Intel(R) Graphics (driver 31.0.101.5186; dedicated VRAM not reported)",
    }


def write_results(run: dict[str, Any]) -> None:
    ARTIFACTS.mkdir(exist_ok=True)
    (ARTIFACTS / "results.json").write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")
    with (ARTIFACTS / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        for row in run["results"]:
            writer.writerow({key: row.get(key) for key in RESULT_COLUMNS})


def run_pilot() -> int:
    started = utc_now()
    models, blocker = discover_models()
    run: dict[str, Any] = {
        "experiment": "002-agent-patch-quality", "pilot": True,
        "started_utc": started, "finished_utc": None,
        "seed": 42, "temperature": 0, "think": False,
        "models_requested": list(MODEL_PREFERENCES),
        "models": [], "environment": hardware_info(),
        "exact_command": f'"{sys.executable}" pilot.py run',
        "blocker": blocker, "results": [],
        "notes": ["Five handcrafted scenarios are an exploratory pilot, not a general benchmark."],
    }
    if blocker:
        for model in MODEL_PREFERENCES:
            for scenario in SCENARIOS:
                run["results"].append({
                    "model": model, "scenario": scenario.id, "category": scenario.category,
                    "status": "blocked", "root_cause_accurate": None, "patch_applicable": None,
                    "failing_test_resolved": None, "regression_free": None,
                    "prompt_tokens": None, "generated_tokens": None, "latency_seconds": None,
                    "error": blocker,
                })
        run["finished_utc"] = utc_now(); write_results(run); write_report(run)
        print(blocker, file=sys.stderr)
        return 2
    raw_dir = ARTIFACTS / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for model in models:
        run["models"].append(model_metadata(model))
        for scenario in SCENARIOS:
            prompt = prompt_for(scenario)
            request_payload = {
                "model": model, "prompt": prompt, "stream": False, "think": False,
                "options": {"temperature": 0, "seed": 42, "num_predict": 1400},
            }
            before = time.perf_counter()
            row = {
                "model": model, "scenario": scenario.id, "category": scenario.category,
                "status": "error", "root_cause_accurate": False, "patch_applicable": False,
                "failing_test_resolved": False, "regression_free": False,
                "prompt_tokens": None, "generated_tokens": None, "latency_seconds": None, "error": "",
            }
            record: dict[str, Any] = {"prompt": prompt, "request": request_payload}
            try:
                response = api("/api/generate", request_payload, timeout=600)
                row["latency_seconds"] = round(time.perf_counter() - before, 6)
                row["prompt_tokens"] = response.get("prompt_eval_count")
                row["generated_tokens"] = response.get("eval_count")
                record["response"] = response
                parsed = parse_model_output(response.get("response", ""))
                row["root_cause_accurate"] = root_cause_score(parsed["root_cause"], scenario.root_cause_terms)
                applicable, public_passing, passing, test_output = evaluate_patch(scenario, parsed["patch"])
                row["patch_applicable"] = applicable
                row["failing_test_resolved"] = public_passing
                row["regression_free"] = passing
                row["status"] = "passed" if passing else "failed"
                record["parsed"] = parsed; record["test_output"] = test_output
            except Exception as exc:
                row["latency_seconds"] = row["latency_seconds"] or round(time.perf_counter() - before, 6)
                row["error"] = f"{type(exc).__name__}: {exc}"
                record["error"] = row["error"]
            (raw_dir / f"{model.replace(':', '_')}-{scenario.id}.json").write_text(
                json.dumps(record, indent=2) + "\n", encoding="utf-8",
            )
            run["results"].append(row); write_results(run)
    run["finished_utc"] = utc_now(); write_results(run); write_report(run)
    return 0


def write_report(run: dict[str, Any]) -> None:
    rows = run["results"]
    lines = [
        "# Experiment 002 pilot report", "",
        "This is a five-scenario handcrafted exploratory pilot, not a general benchmark and not a reproduction of AutoSaddler.", "",
    ]
    if run.get("blocker"):
        lines += ["## Outcome", "", f"Pilot inference was blocked: {run['blocker']}. No model-quality conclusions can be drawn.", ""]
    else:
        lines += ["## Outcome", ""]
        for model in [m["name"] for m in run["models"]]:
            selected = [r for r in rows if r["model"] == model]
            lines.append(f"- {model}: {sum(bool(r['patch_applicable']) for r in selected)}/5 applicable patches; {sum(bool(r['regression_free']) for r in selected)}/5 regression-free fixes.")
        lines.append("")
    lines += [
        "## Limitations", "",
        "The scenarios are synthetic and small, root-cause scoring is keyword-based, and patch success is determined by finite hidden tests. Results do not establish broad model quality or equivalence to AutoSaddler benchmarks.", "",
        "## Reproducibility", "",
        f"Command: `{run['exact_command']}`. Seed: 42. Temperature: 0. Raw prompts and responses are stored under `artifacts/raw/` when inference runs.", "",
    ]
    (ROOT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def self_test() -> int:
    assert len(SCENARIOS) == 5 and len({s.category for s in SCENARIOS}) == 5
    for scenario in SCENARIOS:
        compile(scenario.source, "harness.py", "exec")
        compile(scenario.public_tests, "test_public.py", "exec")
        compile(scenario.hidden_tests, "test_hidden.py", "exec")
        assert scenario.task not in scenario.hidden_tests
        assert "HiddenTests" not in prompt_for(scenario)
    print("pilot self-test: ok (5 scenarios)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Experiment 002 local patch-quality pilot")
    parser.add_argument("command", choices=("run", "self-test"))
    args = parser.parse_args()
    return run_pilot() if args.command == "run" else self_test()


if __name__ == "__main__":
    raise SystemExit(main())
