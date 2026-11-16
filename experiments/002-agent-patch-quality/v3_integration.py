from __future__ import annotations

import argparse
import ast
import copy
import csv
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import v2_integration as v2


ROOT = Path(__file__).resolve().parent
V3_ARTIFACTS = ROOT / "artifacts" / "v3"
TEMP_ROOT = ROOT / ".tmp" / "v3"
MODELS = v2.MODELS
OLLAMA_OPTIONS = v2.OLLAMA_OPTIONS
AUTOSADDLER_COMMIT = v2.AUTOSADDLER_COMMIT


@dataclass(frozen=True, slots=True)
class ScenarioDefinition:
    key: str
    short: str
    function_name: str
    arguments: tuple[str, ...]
    baseline: str
    reference: str
    train_case: str
    development_cases: tuple[str, str]
    task: str
    allowed_imports: tuple[str, ...]
    allowed_calls: tuple[str, ...]
    diagnosis_groups: tuple[tuple[str, ...], ...]
    case_block: str | None = None


SCENARIOS: tuple[ScenarioDefinition, ...] = (
    ScenarioDefinition(
        key="structured_output",
        short="so",
        function_name="extract_result",
        arguments=("response",),
        baseline='''import json

def extract_result(response) -> dict:
    """Return the structured result payload."""
    return dict(response["result"])
''',
        reference='''import json

def extract_result(response) -> dict:
    """Return the structured result payload."""
    value = response["result"]
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        raise ValueError("result must be a JSON object or dict")
    return dict(value)
''',
        train_case="train-json-result-string",
        development_cases=("dev-result-invalid", "dev-result-dict-copy"),
        task="Normalize a structured result payload into an independent dictionary and reject unsupported result values.",
        allowed_imports=("json",),
        allowed_calls=("dict", "isinstance", "str", "ValueError", "json.loads"),
        diagnosis_groups=(("json string", "json-encoded", "decode", "json.loads"), ("dict", "object")),
    ),
    ScenarioDefinition(
        key="tool_arguments",
        short="ta",
        function_name="normalize_arguments",
        arguments=("arguments",),
        baseline=v2.BASELINE_SOURCE,
        reference=v2.REFERENCE_SOURCE,
        train_case="train-json-arguments-string",
        development_cases=("dev-arguments-invalid", "dev-arguments-dict-copy"),
        task="Normalize tool-call arguments into an independent dictionary and reject unsupported argument values.",
        allowed_imports=("json",),
        allowed_calls=("dict", "isinstance", "str", "ValueError", "json.loads"),
        diagnosis_groups=(("json string", "json-encoded", "decode", "json.loads"), ("dict", "object")),
    ),
    ScenarioDefinition(
        key="retries",
        short="rt",
        function_name="should_retry",
        arguments=("status_code", "attempt", "max_attempts"),
        baseline='''def should_retry(status_code, attempt, max_attempts) -> bool:
    """Return whether another attempt should be made."""
    return status_code >= 500 and attempt <= max_attempts
''',
        reference='''def should_retry(status_code, attempt, max_attempts) -> bool:
    """Return whether another attempt should be made."""
    return status_code in (429, 500, 502, 503, 504) and attempt < max_attempts
''',
        train_case="train-retry-budget-exhausted",
        development_cases=("dev-retry-transient", "dev-retry-permanent-boundary"),
        task="Decide whether a transient HTTP failure should be retried without exceeding the attempt budget.",
        allowed_imports=(),
        allowed_calls=(),
        diagnosis_groups=(("off-by-one", "boundary", "exhaust", "attempt <", "<= max"),),
    ),
    ScenarioDefinition(
        key="state_management",
        short="sm",
        function_name="transition_state",
        arguments=("state", "event"),
        baseline='''def transition_state(state, event) -> dict:
    """Return state after applying an event."""
    state["status"] = event
    return state
''',
        reference='''def transition_state(state, event) -> dict:
    """Return state after applying an event."""
    result = dict(state)
    result["status"] = event
    return result
''',
        train_case="train-state-input-immutable",
        development_cases=("dev-state-fields-copy", "dev-state-independent-transitions"),
        task="Apply an event to state while preserving the caller's input and all unrelated fields.",
        allowed_imports=(),
        allowed_calls=("dict",),
        diagnosis_groups=(("mutat", "in-place", "same object", "copy"),),
    ),
    ScenarioDefinition(
        key="error_handling",
        short="eh",
        function_name="safe_error_message",
        arguments=("error",),
        baseline='''def safe_error_message(error) -> str:
    """Return an error message safe for external callers."""
    return str(error)
''',
        reference='''def safe_error_message(error) -> str:
    """Return an error message safe for external callers."""
    return "operation failed"
''',
        train_case="train-error-secret-redaction",
        development_cases=("dev-error-password-redaction", "dev-error-empty-message"),
        task="Return a stable public error message without exposing underlying exception details.",
        allowed_imports=(),
        allowed_calls=("str", "isinstance", "ValueError"),
        diagnosis_groups=(("leak", "secret", "sensitive", "raw error", "str(error)"),),
    ),
)

SCENARIO_BY_KEY = {scenario.key: scenario for scenario in SCENARIOS}


CASE_BLOCKS: Mapping[str, str] = {
    "structured_output": '''if case_id == "train-json-result-string":
    assert target({"result": '{"answer": 42}'}) == {"answer": 42}
elif case_id == "dev-result-invalid":
    for value in ('[1, 2]', None):
        try:
            target({"result": value})
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError for non-object result")
elif case_id == "dev-result-dict-copy":
    original = {"answer": 42, "meta": "kept"}
    result = target({"result": original})
    assert result == original
    assert result is not original
else:
    raise AssertionError("unknown evaluator case")
''',
    "tool_arguments": '''if case_id == "train-json-arguments-string":
    assert target('{"city": "Chicago"}') == {"city": "Chicago"}
elif case_id == "dev-arguments-invalid":
    for value in ('[1, 2]', None):
        try:
            target(value)
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError for non-object arguments")
elif case_id == "dev-arguments-dict-copy":
    original = {"city": "Chicago", "units": "metric"}
    result = target(original)
    assert result == original
    assert result is not original
else:
    raise AssertionError("unknown evaluator case")
''',
    "retries": '''if case_id == "train-retry-budget-exhausted":
    assert target(503, 3, 3) is False
elif case_id == "dev-retry-transient":
    assert target(503, 2, 3) is True
    assert target(429, 1, 3) is True
elif case_id == "dev-retry-permanent-boundary":
    assert target(400, 1, 3) is False
    assert target(401, 1, 3) is False
    assert target(503, 3, 3) is False
else:
    raise AssertionError("unknown evaluator case")
''',
    "state_management": '''if case_id == "train-state-input-immutable":
    original = {"status": "pending", "count": 1}
    result = target(original, "completed")
    assert result == {"status": "completed", "count": 1}
    assert original == {"status": "pending", "count": 1}
    assert result is not original
elif case_id == "dev-state-fields-copy":
    original = {"status": "pending", "owner": "worker-7", "count": 2}
    result = target(original, "running")
    assert result == {"status": "running", "owner": "worker-7", "count": 2}
    assert original["status"] == "pending"
    assert result is not original
elif case_id == "dev-state-independent-transitions":
    original = {"status": "pending", "count": 0}
    first = target(original, "running")
    second = target(original, "failed")
    assert original["status"] == "pending"
    assert first["status"] == "running"
    assert second["status"] == "failed"
    assert first is not second
else:
    raise AssertionError("unknown evaluator case")
''',
    "error_handling": '''if case_id == "train-error-secret-redaction":
    result = target(RuntimeError("database failed: token=TRAIN_SECRET"))
    assert result == "operation failed"
    assert "TRAIN_SECRET" not in result
elif case_id == "dev-error-password-redaction":
    result = target(ValueError("password=DEV_PASSWORD"))
    assert result == "operation failed"
    assert "DEV_PASSWORD" not in result
elif case_id == "dev-error-empty-message":
    assert target(Exception()) == "operation failed"
else:
    raise AssertionError("unknown evaluator case")
''',
}


def validate_component_source(definition: ScenarioDefinition, components: Mapping[str, str]) -> None:
    if set(components) != {"harness.py"}:
        raise ValueError("scenario requires exactly the harness.py component")
    source = components["harness.py"]
    if len(source) > 4000:
        raise ValueError("harness.py exceeds the scenario size limit")
    tree = ast.parse(source, filename="harness.py")
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if len(functions) != 1 or functions[0].name != definition.function_name:
        raise ValueError(f"harness.py must define exactly {definition.function_name}")
    if any(not isinstance(node, (ast.Import, ast.FunctionDef)) for node in tree.body):
        raise ValueError("harness.py may contain only permitted imports and the function definition")
    function = functions[0]
    actual_arguments = tuple(argument.arg for argument in function.args.args)
    if function.decorator_list or actual_arguments != definition.arguments:
        raise ValueError(f"{definition.function_name} must preserve its public signature")
    forbidden_nodes = (
        ast.AsyncFunctionDef,
        ast.Await,
        ast.ClassDef,
        ast.Delete,
        ast.For,
        ast.Global,
        ast.ImportFrom,
        ast.Lambda,
        ast.Nonlocal,
        ast.While,
        ast.With,
        ast.Yield,
        ast.YieldFrom,
    )
    allowed_name_calls = {item for item in definition.allowed_calls if "." not in item}
    for node in ast.walk(tree):
        if isinstance(node, forbidden_nodes):
            raise ValueError(f"forbidden syntax in candidate: {type(node).__name__}")
        if isinstance(node, ast.Import):
            names = tuple(alias.name for alias in node.names)
            if names != definition.allowed_imports:
                raise ValueError(f"candidate imports must be exactly {definition.allowed_imports}")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise ValueError("dunder attribute access is forbidden")
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise ValueError("dunder name access is forbidden")
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id not in allowed_name_calls:
                    raise ValueError(f"forbidden call in candidate: {node.func.id}")
            elif isinstance(node.func, ast.Attribute):
                qualified = (
                    f"{node.func.value.id}.{node.func.attr}"
                    if isinstance(node.func.value, ast.Name)
                    else ""
                )
                if qualified not in definition.allowed_calls and f".{node.func.attr}" not in definition.allowed_calls:
                    raise ValueError("candidate contains a non-allowlisted attribute call")
            else:
                raise ValueError("candidate contains an indirect call")


def runner_source(definition: ScenarioDefinition) -> str:
    return f'''import socket
import sys

class BlockedSocket:
    def __init__(self, *args, **kwargs):
        raise RuntimeError("network disabled in Experiment 002 evaluator")

socket.socket = BlockedSocket
socket.create_connection = BlockedSocket
sys.path.insert(0, ".")
from harness import {definition.function_name} as target

case_id = sys.argv[1]
{definition.case_block if definition.case_block is not None else CASE_BLOCKS[definition.key]}'''


def execute_case(definition: ScenarioDefinition, source: str, case_id: str) -> tuple[bool, str, float]:
    validate_component_source(definition, {"harness.py": source})
    TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    work = TEMP_ROOT / f"c-{uuid.uuid4().hex[:12]}"
    work.mkdir()
    started = time.perf_counter()
    try:
        (work / "harness.py").write_text(source, encoding="utf-8")
        (work / "runner.py").write_text(runner_source(definition), encoding="utf-8")
        child_temp = work / "tmp"
        child_temp.mkdir()
        environment = {
            "PATH": str(Path(sys.executable).parent),
            "PYTHONIOENCODING": "utf-8",
            "PYTHONDONTWRITEBYTECODE": "1",
            "TEMP": str(child_temp),
            "TMP": str(child_temp),
        }
        if os.name == "nt":
            environment["SYSTEMROOT"] = os.environ.get("SYSTEMROOT", r"C:\Windows")
            environment["WINDIR"] = os.environ.get("WINDIR", r"C:\Windows")
        completed = subprocess.run(
            [sys.executable, "-I", "runner.py", case_id],
            cwd=work,
            env=environment,
            capture_output=True,
            text=True,
            timeout=15,
        )
        output = (completed.stdout + completed.stderr).strip().replace(
            str(work), "<isolated-workspace>"
        )
        return completed.returncode == 0, output[-4000:], time.perf_counter() - started
    except subprocess.TimeoutExpired:
        return False, "isolated evaluator timeout", time.perf_counter() - started
    finally:
        if work.resolve().parent == TEMP_ROOT.resolve():
            shutil.rmtree(work, ignore_errors=True)


class ScenarioEvaluator:
    def __init__(
        self,
        *,
        definition: ScenarioDefinition,
        harness_space: v2.ComponentMapHarnessSpace,
        store: v2.LocalRunStore,
    ) -> None:
        self.definition = definition
        self.harness_space = harness_space
        self.store = store
        self.fingerprint = f"exp002-v3-{definition.key}/v1"

    async def evaluate(
        self,
        candidate,
        cases: Sequence[v2.Case],
        context: v2.EvaluationContext,
    ) -> v2.Evaluation:
        materialized = self.harness_space.materialize(candidate, "evaluate")
        try:
            candidate_path = materialized.root / "candidate.json"
            components = json.loads(candidate_path.read_text(encoding="utf-8"))
            source = components["harness.py"]
        finally:
            if os.name == "nt":
                (materialized.root / "candidate.json").chmod(0o666)
            materialized.release()
        observations: list[v2.Observation] = []
        for case in cases:
            for repetition in range(context.repetitions):
                cached = context.attempt_sink.completed(
                    candidate_id=candidate.candidate_id,
                    case_id=case.case_id,
                    repetition=repetition,
                )
                if cached is not None:
                    observations.append(cached)
                    continue
                attempt_id, attempt_number = context.attempt_sink.start(
                    candidate_id=candidate.candidate_id,
                    case_id=case.case_id,
                    repetition=repetition,
                )
                passed, output, wall_seconds = execute_case(self.definition, source, case.case_id)
                trace_relative = (
                    context.artifact_dir / f"{case.case_id}-r{repetition}.json"
                ).relative_to(self.store.run_dir).as_posix()
                trace = self.store.write_json(
                    trace_relative,
                    {
                        "schema_version": "exp002-v3-evaluation-trace/v1",
                        "scenario": self.definition.key,
                        "case_id": case.case_id,
                        "split": context.split,
                        "disposition": "success" if passed else "task_failure",
                        "score": 1.0 if passed else 0.0,
                        "failure_output": output if not passed else "",
                    },
                    kind="exp002-v3-evaluation-trace",
                )
                cost = v2.Cost(rollouts=1, wall_seconds=wall_seconds)
                observation = v2.Observation.create(
                    candidate_id=candidate.candidate_id,
                    case_id=case.case_id,
                    split=case.split,
                    repetition=repetition,
                    disposition="success" if passed else "task_failure",
                    score=1.0 if passed else 0.0,
                    evaluator_fingerprint=self.fingerprint,
                    trace=trace,
                    attempts=attempt_number,
                    cost=cost,
                    metadata={"isolated": True, "scenario": self.definition.key},
                )
                context.attempt_sink.complete(attempt_id, observation, cost)
                observations.append(observation)
        evaluation_id = v2.sha256_digest(context.operation_id)
        prefix = {
            "train": "evaluations",
            "development": "quarantine/dev",
            "test": "post_optimization/test",
        }[context.split]
        return v2.Evaluation(
            evaluation_id=evaluation_id,
            candidate_id=candidate.candidate_id,
            split=context.split,
            purpose=context.purpose,
            iteration=context.iteration,
            requested_case_ids=tuple(case.case_id for case in cases),
            observations=tuple(observations),
            artifact_dir=v2.ArtifactRef(
                uri=f"{prefix}/{evaluation_id.removeprefix('sha256:')}",
                kind="evaluation-directory",
            ),
        )


class ScenarioPromptPack:
    def __init__(self, definition: ScenarioDefinition, store: v2.LocalRunStore) -> None:
        self.definition = definition
        self.store = store
        self.control = v2.ExperimentPromptPack(store)

    def session(self, kind: str, context: Mapping[str, v2.JsonValue]) -> v2.SessionSpec:
        if kind in {"evolve", "reflect"}:
            return self.control.session(kind, context)
        if kind != "diagnose_patch":
            raise ValueError(f"unsupported session kind: {kind}")
        raw_evidence = context.get("evidence")
        if not isinstance(raw_evidence, Mapping) or not isinstance(raw_evidence.get("uri"), str):
            raise TypeError("diagnose_patch requires an evidence artifact")
        evidence = self.store.read_json(raw_evidence["uri"])
        imports = ", ".join(self.definition.allowed_imports) or "none"
        calls = ", ".join(self.definition.allowed_calls) or "none"
        signature = f"{self.definition.function_name}({', '.join(self.definition.arguments)})"
        return v2.SessionSpec(
            kind=kind,
            system_context=(
                "You are the diagnosis-and-modification provider in an AutoSaddler V2 run. "
                "Use only the staged training evidence and current candidate. Hidden development cases exist."
            ),
            task_prompt=(
                f"Scenario: {self.definition.key}. {self.definition.task} "
                f"Diagnose the earliest causal failure in {signature} and propose one general component update. "
                "Return the complete replacement text for harness.py in updates and preserve the public signature. "
                f"Permitted imports: {imports}. Permitted calls: {calls}. "
                "No other imports or calls, file/process/environment/credential/network access, case-specific values, "
                "Markdown, or commentary are allowed. Return exactly the native schema-constrained JSON object."
            ),
            skills={},
            output_schema=v2.DIAGNOSIS_SCHEMA,
            workspace_files={
                ".autosaddler/session_context.json": v2.canonical_json(context) + "\n",
                ".autosaddler/training_evidence.json": v2.canonical_json(evidence) + "\n",
            },
            capabilities=frozenset(),
            mutation_label="capability",
        )


def build_scenario(
    definition: ScenarioDefinition,
    run_dir: Path,
    store: v2.LocalRunStore,
) -> v2.ScenarioComponents:
    harness = v2.ComponentMapHarnessSpace(
        baseline={"harness.py": definition.baseline},
        store_root=run_dir / "candidates",
        materialization_root=run_dir / "materialized",
        validator=lambda components: validate_component_source(definition, components),
    )
    return v2.ScenarioComponents(
        name=f"exp002_v3_{definition.key}",
        version="1",
        harness_space=harness,
        evaluator=ScenarioEvaluator(definition=definition, harness_space=harness, store=store),
        evidence_builder=v2.TrainingEvidenceBuilder(store),
        prompt_pack=ScenarioPromptPack(definition, store),
        train_cases=(v2.Case(case_id=definition.train_case, split="train", payload={}),),
        development_cases=tuple(
            v2.Case(case_id=case_id, split="development", payload={})
            for case_id in definition.development_cases
        ),
        required_capabilities=frozenset(),
        evaluation_repetitions=1,
        resolved_entities={
            "resolved/sources/harness.json": {
                "type": "component_map",
                "baseline_sha256": v2.sha256_digest(definition.baseline),
            },
            "resolved/sources/dataset.json": {
                "train_case_ids": [definition.train_case],
                "development": {
                    "case_ids": list(definition.development_cases),
                    "expectations": "opaque_to_optimizer",
                },
            },
            "resolved/schemas/diagnosis.json": v2.DIAGNOSIS_SCHEMA,
            "resolved/autosaddler_pin.json": {
                "repository": "https://github.com/microsoft/AutoSaddler.git",
                "commit": AUTOSADDLER_COMMIT,
            },
        },
    )


def build_engine(
    definition: ScenarioDefinition,
    run_dir: Path,
    run_id: str,
    provider: v2.HybridOllamaProvider,
) -> tuple[v2.AutoSaddlerEngine, v2.LocalRunStore]:
    store = v2.ProjectionRetryStore(run_dir=run_dir, run_id=run_id)
    scenario = build_scenario(definition, run_dir, store)
    resolved_config: Mapping[str, v2.JsonValue] = {
        "schema_version": "autosaddler/v2",
        "scenario": {"type": f"exp002_v3_{definition.key}", "version": "1"},
        "optimization": {"max_iterations": 1, "max_rollouts": 10, "session_retries": 0},
        "provider": {
            "type": "ollama-exp002",
            "model": provider.model,
            "diagnosis_only": True,
            "options": dict(OLLAMA_OPTIONS),
        },
        "storage": {"type": "local", "run_root": str(run_dir.parent)},
    }
    store.initialize(resolved_config=resolved_config, resolved_entities=scenario.resolved_entities)
    policies = v2.PolicyBundle(
        task_selection=v2.FixedTaskSelectionPolicy(batch_size=1),
        acceptance=v2.MatchedValidStrictImprovement(),
        development=v2.FullOnAcceptDevelopment(),
        ranking=v2.MeanDevelopmentRanking(),
        budget=v2.BudgetPolicy(max_rollouts=10, max_iterations=1),
    )
    engine = v2.AutoSaddlerEngine(
        store=store,
        scenario=scenario,
        provider=provider,
        policies=policies,
        diagnosis_patch_timeout_seconds=360,
        selection_timeout_seconds=30,
        reflection_timeout_seconds=30,
        session_retries=0,
    )
    return engine, store


def diagnosis_accurate(definition: ScenarioDefinition, diagnosis: str) -> bool:
    normalized = diagnosis.casefold()
    return all(any(term.casefold() in normalized for term in group) for group in definition.diagnosis_groups)


def summarize_run(
    definition: ScenarioDefinition,
    model: str,
    store: v2.LocalRunStore,
    provider: v2.HybridOllamaProvider,
) -> dict[str, Any]:
    summary = v2.summarize_run(model, store, provider)
    diagnosis = ""
    if provider.raw_records:
        raw_content = provider.raw_records[0].get("raw_content", "")
        if isinstance(raw_content, str):
            try:
                parsed = json.loads(raw_content)
            except json.JSONDecodeError:
                parsed = {}
            if isinstance(parsed, Mapping) and isinstance(parsed.get("diagnosis"), str):
                diagnosis = parsed["diagnosis"]
    if summary["mutation_rejection"] is not None:
        outcome = "rejected_verifier"
    elif summary["training_score_after"] is None:
        outcome = "unevaluated"
    elif summary["accepted_mutation"]:
        outcome = "accepted"
    else:
        outcome = "evaluated_failure"
    return {
        "scenario": definition.key,
        "diagnosis_accurate": diagnosis_accurate(definition, diagnosis),
        "diagnosis": diagnosis,
        "outcome": outcome,
        **summary,
    }


def preregistration_document() -> dict[str, Any]:
    return {
        "schema_version": "exp002-v3-preregistration/v1",
        "frozen_before_model_execution": True,
        "official_engine": {
            "repository": "https://github.com/microsoft/AutoSaddler.git",
            "commit": AUTOSADDLER_COMMIT,
            "core_modified": False,
        },
        "models": list(MODELS),
        "model_settings": {**dict(OLLAMA_OPTIONS), "think": False, "session_retries": 0},
        "workflow": [
            "seed development baseline",
            "fixed training selection",
            "training evidence",
            "diagnose_patch structured update",
            "component verification",
            "matched valid strict training improvement",
            "full development on accept",
            "mean development ranking",
        ],
        "metrics": [
            "diagnosis_accuracy_fixed_concept_groups",
            "schema_validity",
            "mutation_validity",
            "training_improvement",
            "accepted_mutation",
            "development_improvement",
            "development_regressions",
            "prompt_tokens",
            "generated_tokens",
            "ollama_latency_seconds",
        ],
        "isolation": {
            "network": "socket creation blocked in stripped-environment subprocess",
            "filesystem": "unique disposable working directory plus AST operation allowlist",
            "timeout_seconds": 15,
            "traceback_path_sanitizer": "replace exact disposable root with <isolated-workspace>",
        },
        "scenarios": [
            {
                "key": item.key,
                "function": item.function_name,
                "arguments": list(item.arguments),
                "task": item.task,
                "train_case_ids": [item.train_case],
                "development_case_ids": list(item.development_cases),
                "baseline_sha256": v2.sha256_digest(item.baseline),
                "reference_sha256": v2.sha256_digest(item.reference),
                "case_runner_sha256": v2.sha256_digest(runner_source(item)),
                "allowed_imports": list(item.allowed_imports),
                "allowed_calls": list(item.allowed_calls),
                "forbidden": [
                    "unlisted imports or calls",
                    "dunder access",
                    "async/class/loop/with/lambda/yield syntax",
                    "file/process/environment/credential/network access",
                    "signature or component-schema changes",
                ],
                "diagnosis_concept_groups": [list(group) for group in item.diagnosis_groups],
            }
            for item in SCENARIOS
        ],
    }


def preregister() -> int:
    if (V3_ARTIFACTS / "runs").exists() or (V3_ARTIFACTS / "raw").exists():
        raise RuntimeError("refusing to preregister after V3 model artifacts exist")
    V3_ARTIFACTS.mkdir(parents=True, exist_ok=True)
    path = V3_ARTIFACTS / "preregistration.json"
    if path.exists():
        raise RuntimeError(f"refusing to overwrite preregistration: {path}")
    payload = v2.canonical_json(preregistration_document()) + "\n"
    path.write_text(payload, encoding="utf-8")
    digest = v2.sha256_digest(payload.rstrip("\n"))
    (V3_ARTIFACTS / "preregistration.sha256").write_text(digest + "\n", encoding="utf-8")
    print(f"V3 preregistered: {digest}")
    return 0


def verify_preregistration() -> str:
    path = V3_ARTIFACTS / "preregistration.json"
    digest_path = V3_ARTIFACTS / "preregistration.sha256"
    if not path.is_file() or not digest_path.is_file():
        raise FileNotFoundError("V3 preregistration artifacts are missing")
    expected = v2.canonical_json(preregistration_document()) + "\n"
    actual = path.read_text(encoding="utf-8")
    if actual != expected:
        raise RuntimeError("V3 preregistration no longer matches the implementation")
    digest = v2.sha256_digest(actual.rstrip("\n"))
    if digest_path.read_text(encoding="utf-8").strip() != digest:
        raise RuntimeError("V3 preregistration digest mismatch")
    return digest


def model_slug(model: str) -> str:
    return "q17" if model == "qwen3:1.7b" else "q4"


def run_one(definition: ScenarioDefinition, model: str) -> dict[str, Any]:
    run_id = f"exp002-v3-{definition.key}-{model.replace(':', '-').replace('.', '_')}"
    run_dir = V3_ARTIFACTS / "runs" / definition.short / model_slug(model)
    if run_dir.exists():
        raise RuntimeError(f"refusing to overwrite existing V3 run: {run_dir}")
    provider = v2.HybridOllamaProvider(model)
    engine, store = build_engine(definition, run_dir, run_id, provider)
    engine.run()
    raw_dir = V3_ARTIFACTS / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / f"{definition.short}-{model_slug(model)}.json"
    raw_path.write_text(json.dumps(provider.raw_records, indent=2) + "\n", encoding="utf-8")
    return summarize_run(definition, model, store, provider)


def run_all() -> int:
    preregistration_digest = verify_preregistration()
    if (V3_ARTIFACTS / "results.json").exists():
        raise RuntimeError("refusing to overwrite completed V3 results")
    results: list[dict[str, Any]] = []
    for definition in SCENARIOS:
        for model in MODELS:
            results.append(run_one(definition, model))
    document = {
        "schema_version": "exp002-v3-results/v1",
        "autosaddler_commit": AUTOSADDLER_COMMIT,
        "preregistration_sha256": preregistration_digest,
        "results": results,
    }
    (V3_ARTIFACTS / "results.json").write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    fieldnames = [
        "scenario", "model", "diagnosis_accurate", "schema_valid", "mutation_finalized",
        "mutation_rejection", "outcome", "accepted_mutation", "training_score_before",
        "training_score_after", "training_improvement", "development_score_before",
        "development_score_after", "development_evaluated", "development_improvement",
        "development_regressions", "prompt_tokens", "generated_tokens", "latency_seconds",
    ]
    with (V3_ARTIFACTS / "results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in results:
            serialized = dict(row)
            serialized["development_regressions"] = json.dumps(row["development_regressions"])
            writer.writerow(serialized)
    print(json.dumps(document, indent=2))
    return 0


def normalized_request(record: Mapping[str, Any]) -> Mapping[str, Any]:
    request = copy.deepcopy(record["request"])
    request["model"] = "MODEL_UNDER_TEST"
    return request


def audit_results() -> int:
    results_path = V3_ARTIFACTS / "results.json"
    if not results_path.is_file():
        raise FileNotFoundError("V3 results are missing")
    document = json.loads(results_path.read_text(encoding="utf-8"))
    rows = document["results"]
    checks: list[dict[str, Any]] = []
    for definition in SCENARIOS:
        scenario_rows = [row for row in rows if row["scenario"] == definition.key]
        if len(scenario_rows) != 2:
            raise RuntimeError(f"expected two rows for {definition.key}")
        records = []
        for model in MODELS:
            raw_path = V3_ARTIFACTS / "raw" / f"{definition.short}-{model_slug(model)}.json"
            loaded = json.loads(raw_path.read_text(encoding="utf-8"))
            if len(loaded) != 1:
                raise RuntimeError(f"expected one raw diagnosis call for {definition.key}/{model}")
            records.append(loaded[0])
            row = next(item for item in scenario_rows if item["model"] == model)
            response = loaded[0].get("response", {})
            if row["prompt_tokens"] != response.get("prompt_eval_count"):
                raise RuntimeError(f"prompt token mismatch for {definition.key}/{model}")
            if row["generated_tokens"] != response.get("eval_count"):
                raise RuntimeError(f"generated token mismatch for {definition.key}/{model}")
            run_dir = V3_ARTIFACTS / "runs" / definition.short / model_slug(model)
            events = v2.LocalRunStore(
                run_dir=run_dir,
                run_id=f"exp002-v3-{definition.key}-{model.replace(':', '-').replace('.', '_')}",
            ).events()
            diagnosis_sessions = [
                event for event in events
                if event.event_type in {"SessionCompleted", "SessionFailed"}
                and "diagnose-patch" in str(event.payload.get("logical_operation_id", ""))
            ]
            if len(diagnosis_sessions) != 1:
                raise RuntimeError(f"diagnosis session count mismatch for {definition.key}/{model}")
        identical = v2.canonical_json(normalized_request(records[0])) == v2.canonical_json(
            normalized_request(records[1])
        )
        if not identical:
            raise RuntimeError(f"model-facing requests differ for {definition.key}")
        checks.append({"scenario": definition.key, "byte_identical_normalized_requests": True})
    audit = {
        "schema_version": "exp002-v3-audit/v1",
        "preregistration_verified": verify_preregistration(),
        "raw_rows": len(rows),
        "checks": checks,
    }
    (V3_ARTIFACTS / "audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Experiment 002 five-scenario AutoSaddler V2 comparison")
    parser.add_argument("command", choices=("preregister", "run", "audit", "check-pin"))
    args = parser.parse_args()
    if args.command == "preregister":
        return preregister()
    if args.command == "run":
        return run_all()
    if args.command == "audit":
        return audit_results()
    actual = v2.main_check_pin_for_test()
    print(f"AutoSaddler pin verified: {actual}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
