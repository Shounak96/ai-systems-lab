from __future__ import annotations

import argparse
import ast
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
import uuid
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
AUTOSADDLER_ROOT = ROOT / "vendor" / "AutoSaddler"
AUTOSADDLER_SRC = AUTOSADDLER_ROOT / "src"
AUTOSADDLER_COMMIT = "e841543ce6cf6bdf690e65134b12a829ec9f66d1"
if not AUTOSADDLER_SRC.is_dir():
    raise RuntimeError(f"Pinned AutoSaddler source is missing: {AUTOSADDLER_SRC}")
sys.path.insert(0, str(AUTOSADDLER_SRC))

from autosaddler.v2.core.domain import (  # noqa: E402
    ArtifactRef,
    Case,
    Cost,
    Evaluation,
    JsonValue,
    Observation,
    canonical_json,
    sha256_digest,
)
from autosaddler.v2.core.engine import AutoSaddlerEngine  # noqa: E402
from autosaddler.v2.core.policies import (  # noqa: E402
    BudgetPolicy,
    FixedTaskSelectionPolicy,
    FullOnAcceptDevelopment,
    MatchedValidStrictImprovement,
    MeanDevelopmentRanking,
    PolicyBundle,
)
from autosaddler.v2.core.ports import EvaluationContext, ScenarioComponents  # noqa: E402
from autosaddler.v2.core.serde import evaluation_from  # noqa: E402
from autosaddler.v2.harness.component_map import ComponentMapHarnessSpace  # noqa: E402
from autosaddler.v2.prompting.models import (  # noqa: E402
    SessionRequest,
    SessionResult,
    SessionSpec,
    Usage,
)
from autosaddler.v2.storage.local import LocalRunStore  # noqa: E402


class ProjectionRetryStore(LocalRunStore):
    """Retry only transient Windows projection replacements; event writes are never replayed."""

    projection_retry_limit = 8

    def refresh_projections(self) -> None:
        for attempt in range(self.projection_retry_limit):
            try:
                super().refresh_projections()
                return
            except PermissionError:
                if attempt + 1 == self.projection_retry_limit:
                    raise
                time.sleep(0.025 * (attempt + 1))


V2_ARTIFACTS = ROOT / "artifacts" / "v2"
TEMP_ROOT = ROOT / ".tmp"
MODELS = ("qwen3:1.7b", "qwen3:4b")
BASELINE_SOURCE = '''import json

def normalize_arguments(arguments) -> dict:
    """Return normalized tool-call arguments."""
    return dict(arguments)
'''
REFERENCE_SOURCE = '''import json

def normalize_arguments(arguments) -> dict:
    """Return normalized tool-call arguments."""
    if isinstance(arguments, str):
        arguments = json.loads(arguments)
    if not isinstance(arguments, dict):
        raise ValueError("arguments must be a JSON object or dict")
    return dict(arguments)
'''
TRAIN_CASE_ID = "train-json-object-string"
DEV_CASE_IDS = ("dev-invalid-values", "dev-dict-preservation")
OLLAMA_OPTIONS: Mapping[str, JsonValue] = {
    "temperature": 0,
    "seed": 42,
    "num_predict": 1200,
    "num_ctx": 4096,
}


DIAGNOSIS_SCHEMA: Mapping[str, JsonValue] = {
    "type": "object",
    "required": ["schema_version", "diagnosis", "expected_effect", "updates"],
    "properties": {
        "schema_version": {"const": "exp002-diagnosis-update/v1"},
        "diagnosis": {"type": "string", "minLength": 1},
        "expected_effect": {"type": "string", "minLength": 1},
        "updates": {
            "type": "object",
            "required": ["harness.py"],
            "properties": {"harness.py": {"type": "string", "minLength": 1, "maxLength": 4000}},
            "additionalProperties": False,
        },
    },
    "additionalProperties": False,
}


def _evolve_schema(candidate_ids: Sequence[str]) -> Mapping[str, JsonValue]:
    return {
        "type": "object",
        "required": ["schema_version", "parent_ids", "component_sources", "rationale"],
        "properties": {
            "schema_version": {"const": "exp002-evolution/v1"},
            "parent_ids": {
                "type": "array",
                "items": {"type": "string", "enum": list(candidate_ids)},
                "minItems": 1,
                "uniqueItems": True,
            },
            "component_sources": {"type": "object", "additionalProperties": False},
            "rationale": {"type": "string", "minLength": 1},
        },
        "additionalProperties": False,
    }


REFLECTION_SCHEMA: Mapping[str, JsonValue] = {
    "type": "object",
    "required": ["schema_version", "lessons"],
    "properties": {
        "schema_version": {"const": "exp002-reflection/v1"},
        "lessons": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["scope", "statement", "evidence_case_ids"],
                "properties": {
                    "scope": {"type": "string", "minLength": 1},
                    "statement": {"type": "string", "minLength": 1},
                    "evidence_case_ids": {"type": "array", "items": {"type": "string"}},
                },
                "additionalProperties": False,
            },
        },
    },
    "additionalProperties": False,
}


def validate_component_source(components: Mapping[str, str]) -> None:
    if set(components) != {"harness.py"}:
        raise ValueError("scenario requires exactly the harness.py component")
    source = components["harness.py"]
    if len(source) > 4000:
        raise ValueError("harness.py exceeds the scenario size limit")
    tree = ast.parse(source, filename="harness.py")
    top_level_functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if len(top_level_functions) != 1 or top_level_functions[0].name != "normalize_arguments":
        raise ValueError("harness.py must define exactly normalize_arguments")
    if any(not isinstance(node, (ast.Import, ast.FunctionDef)) for node in tree.body):
        raise ValueError("harness.py may contain only the json import and function definition")
    function = top_level_functions[0]
    if function.decorator_list or len(function.args.args) != 1 or function.args.args[0].arg != "arguments":
        raise ValueError("normalize_arguments must keep its one-argument public API")
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
    for node in ast.walk(tree):
        if isinstance(node, forbidden_nodes):
            raise ValueError(f"forbidden syntax in candidate: {type(node).__name__}")
        if isinstance(node, ast.Import):
            if [alias.name for alias in node.names] != ["json"]:
                raise ValueError("candidate may import only json")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise ValueError("dunder attribute access is forbidden")
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise ValueError("dunder name access is forbidden")
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id not in {"dict", "isinstance", "str", "ValueError"}:
                    raise ValueError(f"forbidden call in candidate: {node.func.id}")
            elif isinstance(node.func, ast.Attribute):
                is_json_loads = (
                    isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "json"
                    and node.func.attr == "loads"
                )
                is_safe_argument_method = (
                    isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "arguments"
                    and node.func.attr in {"copy", "strip"}
                )
                if not (is_json_loads or is_safe_argument_method):
                    raise ValueError("candidate contains a non-allowlisted attribute call")
            else:
                raise ValueError("candidate contains an indirect call")


ISOLATED_RUNNER = r'''import socket
import sys

class BlockedSocket:
    def __init__(self, *args, **kwargs):
        raise RuntimeError("network disabled in Experiment 002 evaluator")

socket.socket = BlockedSocket
socket.create_connection = BlockedSocket
sys.path.insert(0, ".")
from harness import normalize_arguments

case_id = sys.argv[1]
if case_id == "train-json-object-string":
    assert normalize_arguments('{"city": "Chicago"}') == {"city": "Chicago"}
elif case_id == "dev-invalid-values":
    for value in ('[1, 2]', '{bad', None):
        try:
            normalize_arguments(value)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {type(value).__name__}")
elif case_id == "dev-dict-preservation":
    original = {"city": "Chicago"}
    result = normalize_arguments(original)
    assert result == original
    assert result is not original
else:
    raise AssertionError("unknown evaluator case")
'''


def execute_case(source: str, case_id: str) -> tuple[bool, str, float]:
    validate_component_source({"harness.py": source})
    TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    work = TEMP_ROOT / f"exp002-v2-{uuid.uuid4().hex}"
    work.mkdir()
    started = time.perf_counter()
    try:
        (work / "harness.py").write_text(source, encoding="utf-8")
        (work / "runner.py").write_text(ISOLATED_RUNNER, encoding="utf-8")
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


class DeterministicHarnessEvaluator:
    fingerprint = "exp002-tool-arguments/v2"

    def __init__(self, *, harness_space: ComponentMapHarnessSpace, store: LocalRunStore) -> None:
        self.harness_space = harness_space
        self.store = store

    async def evaluate(
        self,
        candidate,
        cases: Sequence[Case],
        context: EvaluationContext,
    ) -> Evaluation:
        materialized = self.harness_space.materialize(candidate, "evaluate")
        try:
            candidate_path = materialized.root / "candidate.json"
            components = json.loads(candidate_path.read_text(encoding="utf-8"))
            source = components["harness.py"]
        finally:
            # ComponentMapHarnessSpace uses chmod(0444). On Windows that maps to
            # the read-only attribute, which must be cleared before its release
            # callback can remove the disposable materialization directory.
            if os.name == "nt":
                (materialized.root / "candidate.json").chmod(0o666)
            materialized.release()
        observations: list[Observation] = []
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
                passed, output, wall_seconds = execute_case(source, case.case_id)
                trace_relative = (
                    context.artifact_dir / f"{case.case_id}-r{repetition}.json"
                ).relative_to(self.store.run_dir).as_posix()
                trace = self.store.write_json(
                    trace_relative,
                    {
                        "schema_version": "exp002-evaluation-trace/v1",
                        "case_id": case.case_id,
                        "split": context.split,
                        "disposition": "success" if passed else "task_failure",
                        "score": 1.0 if passed else 0.0,
                        "failure_output": output if not passed else "",
                    },
                    kind="exp002-evaluation-trace",
                )
                cost = Cost(rollouts=1, wall_seconds=wall_seconds)
                observation = Observation.create(
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
                    metadata={"isolated": True},
                )
                context.attempt_sink.complete(attempt_id, observation, cost)
                observations.append(observation)
        evaluation_id = sha256_digest(context.operation_id)
        prefix = {
            "train": "evaluations",
            "development": "quarantine/dev",
            "test": "post_optimization/test",
        }[context.split]
        return Evaluation(
            evaluation_id=evaluation_id,
            candidate_id=candidate.candidate_id,
            split=context.split,
            purpose=context.purpose,
            iteration=context.iteration,
            requested_case_ids=tuple(case.case_id for case in cases),
            observations=tuple(observations),
            artifact_dir=ArtifactRef(
                uri=f"{prefix}/{evaluation_id.removeprefix('sha256:')}",
                kind="evaluation-directory",
            ),
        )


class TrainingEvidenceBuilder:
    def __init__(self, store: LocalRunStore) -> None:
        self.store = store

    def build(self, evaluation: Evaluation) -> ArtifactRef:
        if evaluation.split != "train":
            raise ValueError("diagnosis evidence must be built from training only")
        records = []
        for observation in evaluation.observations:
            trace = self.store.read_json(observation.trace.uri) if observation.trace else {}
            records.append(
                {
                    "case_id": observation.case_id,
                    "disposition": observation.disposition,
                    "score": observation.score,
                    "failure_trace": trace,
                }
            )
        return self.store.write_json(
            f"evidence/{evaluation.evaluation_id.removeprefix('sha256:')}/evidence.json",
            {
                "schema_version": "exp002-training-evidence/v1",
                "candidate_id": evaluation.candidate_id,
                "split": "train",
                "case_records": records,
            },
            kind="exp002-training-evidence",
        )


class ExperimentPromptPack:
    def __init__(self, store: LocalRunStore) -> None:
        self.store = store

    def session(self, kind: str, context: Mapping[str, JsonValue]) -> SessionSpec:
        if kind == "evolve":
            candidate_ids = context.get("candidate_ids")
            if not isinstance(candidate_ids, list) or not all(isinstance(item, str) for item in candidate_ids):
                raise TypeError("evolve requires candidate_ids")
            return SessionSpec(
                kind=kind,
                system_context="Deterministic control stage.",
                task_prompt="Select the only seed candidate.",
                skills={},
                output_schema=_evolve_schema(candidate_ids),
                workspace_files={".autosaddler/control_context.json": canonical_json(context) + "\n"},
                capabilities=frozenset(),
            )
        if kind == "reflect":
            return SessionSpec(
                kind=kind,
                system_context="Deterministic control stage.",
                task_prompt="Record the measured outcome without additional model inference.",
                skills={},
                output_schema=REFLECTION_SCHEMA,
                workspace_files={".autosaddler/control_context.json": canonical_json(context) + "\n"},
                capabilities=frozenset(),
            )
        if kind != "diagnose_patch":
            raise ValueError(f"unsupported session kind: {kind}")
        raw_evidence = context.get("evidence")
        if not isinstance(raw_evidence, Mapping) or not isinstance(raw_evidence.get("uri"), str):
            raise TypeError("diagnose_patch requires an evidence artifact")
        evidence = self.store.read_json(raw_evidence["uri"])
        return SessionSpec(
            kind=kind,
            system_context=(
                "You are the diagnosis-and-modification provider in an AutoSaddler V2 run. "
                "Use only the staged training evidence and current candidate. Hidden development cases exist."
            ),
            task_prompt=(
                "Diagnose the earliest causal failure in normalize_arguments and propose one general component "
                "update. Return the complete replacement text for harness.py in updates. Preserve its public API. "
                "The verifier permits only `import json`; calls to json.loads, dict, isinstance, str, and ValueError; "
                "and no file, process, environment, credential, or network access. Do not encode case-specific values. "
                "Return exactly the native schema-constrained JSON object, with no Markdown or commentary."
            ),
            skills={},
            output_schema=DIAGNOSIS_SCHEMA,
            workspace_files={
                ".autosaddler/session_context.json": canonical_json(context) + "\n",
                ".autosaddler/training_evidence.json": canonical_json(evidence) + "\n",
            },
            capabilities=frozenset(),
            mutation_label="capability",
        )


class HybridOllamaProvider:
    def __init__(
        self,
        model: str,
        *,
        diagnosis_override: Callable[[SessionRequest], Mapping[str, JsonValue]] | None = None,
    ) -> None:
        self.model = model
        self.diagnosis_override = diagnosis_override
        self.diagnosis_calls = 0
        self.raw_records: list[Mapping[str, Any]] = []

    async def run(self, request: SessionRequest) -> SessionResult:
        if request.spec.kind == "evolve":
            context = json.loads(request.spec.workspace_files[".autosaddler/control_context.json"])
            response: Mapping[str, JsonValue] = {
                "schema_version": "exp002-evolution/v1",
                "parent_ids": [context["candidate_ids"][0]],
                "component_sources": {},
                "rationale": "Use the only accepted seed candidate.",
            }
            return _deterministic_result(response)
        if request.spec.kind == "reflect":
            context = json.loads(request.spec.workspace_files[".autosaddler/control_context.json"])
            response = {
                "schema_version": "exp002-reflection/v1",
                "lessons": [
                    {
                        "scope": "component",
                        "statement": "Outcome recorded deterministically to isolate diagnosis quality.",
                        "evidence_case_ids": context["train_case_ids"],
                    }
                ],
            }
            return _deterministic_result(response)
        if request.spec.kind != "diagnose_patch":
            raise ValueError(f"unsupported provider session: {request.spec.kind}")
        self.diagnosis_calls += 1
        if self.diagnosis_calls != 1:
            raise RuntimeError("only one diagnosis model call is permitted per run")
        if self.diagnosis_override is not None:
            response = self.diagnosis_override(request)
            return _deterministic_result(response, model="reference-integration-test")
        return self._run_ollama(request)

    def _run_ollama(self, request: SessionRequest) -> SessionResult:
        payload = self.build_ollama_payload(request)
        started = time.perf_counter()
        raw_response = ""
        ollama_response: Mapping[str, Any] = {}
        try:
            encoded = json.dumps(payload).encode("utf-8")
            http_request = urllib.request.Request(
                "http://127.0.0.1:11434/api/chat",
                data=encoded,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(http_request, timeout=request.timeout_seconds) as response:
                loaded = json.load(response)
            if not isinstance(loaded, dict):
                raise TypeError("Ollama response must be an object")
            ollama_response = loaded
            message = loaded.get("message")
            if not isinstance(message, Mapping) or not isinstance(message.get("content"), str):
                raise TypeError("Ollama response is missing message.content")
            raw_response = message["content"]
            parsed = json.loads(raw_response)
            structured = parsed if isinstance(parsed, dict) else None
            usage = Usage(
                role="optimizer",
                input_tokens=int(loaded.get("prompt_eval_count", 0) or 0),
                output_tokens=int(loaded.get("eval_count", 0) or 0),
                model=self.model,
                duration_seconds=time.perf_counter() - started,
                configured_settings={"think": False, **dict(OLLAMA_OPTIONS)},
                provider_metadata={"done_reason": str(loaded.get("done_reason", ""))},
            )
            self.raw_records.append(
                {"request": payload, "response": loaded, "raw_content": raw_response}
            )
            return SessionResult(
                status="completed",
                structured_output=structured,
                raw_response=raw_response,
                tool_calls=(),
                usage=(usage,),
                cost=Cost(
                    sessions=1,
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                    wall_seconds=usage.duration_seconds or 0.0,
                ),
            )
        except Exception as error:
            elapsed = time.perf_counter() - started
            self.raw_records.append(
                {
                    "request": payload,
                    "response": ollama_response,
                    "raw_content": raw_response,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            usage = Usage(
                role="optimizer",
                input_tokens=int(ollama_response.get("prompt_eval_count", 0) or 0),
                output_tokens=int(ollama_response.get("eval_count", 0) or 0),
                model=self.model,
                duration_seconds=elapsed,
                status="failed",
                error_type=type(error).__name__,
                usage_incomplete=not bool(ollama_response),
            )
            return SessionResult(
                status="failed",
                structured_output=None,
                raw_response=raw_response,
                tool_calls=(),
                usage=(usage,),
                cost=Cost(
                    sessions=1,
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                    wall_seconds=elapsed,
                ),
                error=f"{type(error).__name__}: {error}",
            )

    def build_ollama_payload(self, request: SessionRequest) -> dict[str, Any]:
        candidate = json.loads((request.workspace / "candidate.json").read_text(encoding="utf-8"))
        evidence = json.loads(request.spec.workspace_files[".autosaddler/training_evidence.json"])
        user_prompt = (
            f"{request.spec.task_prompt}\n\nCURRENT CANDIDATE:\n{canonical_json(candidate)}"
            f"\n\nTRAINING EVIDENCE:\n{canonical_json(evidence)}"
        )
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": request.spec.system_context},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "think": False,
            "format": dict(request.spec.output_schema),
            "options": dict(OLLAMA_OPTIONS),
        }
        return payload


def _deterministic_result(
    response: Mapping[str, JsonValue],
    *,
    model: str = "deterministic-control-v1",
) -> SessionResult:
    usage = Usage(role="control", model=model)
    return SessionResult(
        status="completed",
        structured_output=response,
        raw_response=canonical_json(response),
        tool_calls=(),
        usage=(usage,),
        cost=Cost(sessions=1),
    )


def build_scenario(run_dir: Path, store: LocalRunStore) -> ScenarioComponents:
    harness = ComponentMapHarnessSpace(
        baseline={"harness.py": BASELINE_SOURCE},
        store_root=run_dir / "candidates",
        materialization_root=run_dir / "materialized",
        validator=validate_component_source,
    )
    return ScenarioComponents(
        name="exp002_tool_arguments",
        version="1",
        harness_space=harness,
        evaluator=DeterministicHarnessEvaluator(harness_space=harness, store=store),
        evidence_builder=TrainingEvidenceBuilder(store),
        prompt_pack=ExperimentPromptPack(store),
        train_cases=(Case(case_id=TRAIN_CASE_ID, split="train", payload={}),),
        development_cases=tuple(Case(case_id=case_id, split="development", payload={}) for case_id in DEV_CASE_IDS),
        required_capabilities=frozenset(),
        evaluation_repetitions=1,
        resolved_entities={
            "resolved/sources/harness.json": {
                "type": "component_map",
                "baseline_sha256": sha256_digest(BASELINE_SOURCE),
            },
            "resolved/sources/dataset.json": {
                "train_case_ids": [TRAIN_CASE_ID],
                "development": {"case_ids": list(DEV_CASE_IDS), "expectations": "opaque_to_optimizer"},
            },
            "resolved/schemas/diagnosis.json": DIAGNOSIS_SCHEMA,
            "resolved/autosaddler_pin.json": {
                "repository": "https://github.com/microsoft/AutoSaddler.git",
                "commit": AUTOSADDLER_COMMIT,
            },
        },
    )


def build_engine(
    run_dir: Path,
    run_id: str,
    provider: HybridOllamaProvider,
) -> tuple[AutoSaddlerEngine, LocalRunStore]:
    store = ProjectionRetryStore(run_dir=run_dir, run_id=run_id)
    scenario = build_scenario(run_dir, store)
    resolved_config: Mapping[str, JsonValue] = {
        "schema_version": "autosaddler/v2",
        "scenario": {"type": "exp002_tool_arguments", "version": "1"},
        "optimization": {
            "max_iterations": 1,
            "max_rollouts": 10,
            "session_retries": 0,
        },
        "provider": {
            "type": "ollama-exp002",
            "model": provider.model,
            "diagnosis_only": True,
            "options": dict(OLLAMA_OPTIONS),
        },
        "storage": {"type": "local", "run_root": str(run_dir.parent)},
    }
    store.initialize(resolved_config=resolved_config, resolved_entities=scenario.resolved_entities)
    policies = PolicyBundle(
        task_selection=FixedTaskSelectionPolicy(batch_size=1),
        acceptance=MatchedValidStrictImprovement(),
        development=FullOnAcceptDevelopment(),
        ranking=MeanDevelopmentRanking(),
        budget=BudgetPolicy(max_rollouts=10, max_iterations=1),
    )
    return (
        AutoSaddlerEngine(
            store=store,
            scenario=scenario,
            provider=provider,
            policies=policies,
            diagnosis_patch_timeout_seconds=360,
            selection_timeout_seconds=30,
            reflection_timeout_seconds=30,
            session_retries=0,
        ),
        store,
    )


def summarize_run(model: str, store: LocalRunStore, provider: HybridOllamaProvider) -> dict[str, Any]:
    events = store.events()
    diagnosis_events = [
        event
        for event in events
        if event.event_type in {"SessionCompleted", "SessionFailed"}
        and "diagnose-patch" in str(event.payload.get("logical_operation_id", ""))
    ]
    diagnosis_completed = any(event.event_type == "SessionCompleted" for event in diagnosis_events)
    finalized_events = [event for event in events if event.event_type == "CandidateFinalized"]
    mutation_finalized = any(event.payload.get("kind") != "composition" for event in finalized_events)
    rejection_events = [event for event in events if event.event_type == "MutationRejected"]
    mutation_rejection = (
        rejection_events[-1].payload.get("verification_failure") if rejection_events else None
    )
    acceptance_events = [event for event in events if event.event_type == "AcceptanceDecided"]
    verdict = acceptance_events[-1].payload.get("verdict", {}) if acceptance_events else {}
    accepted = isinstance(verdict, Mapping) and verdict.get("accepted") is True
    evaluations = [
        evaluation_from(event.payload.get("evaluation"))
        for event in events
        if event.event_type == "EvaluationCompleted"
    ]
    training_before = next((item.aggregate_score for item in evaluations if item.purpose == "train_before"), None)
    training_after = next((item.aggregate_score for item in evaluations if item.purpose == "train_after"), None)
    development = [item for item in evaluations if item.purpose == "development"]
    development_before = development[0].aggregate_score if development else None
    development_after = development[-1].aggregate_score if len(development) > 1 else None
    regressions: list[str] | None = None
    if len(development) > 1:
        before_scores = {item.case_id: item.score for item in development[0].observations}
        after_scores = {item.case_id: item.score for item in development[-1].observations}
        regressions = [
            case_id
            for case_id, before in before_scores.items()
            if before is not None and before > 0 and (after_scores.get(case_id) or 0) < before
        ]
    raw = provider.raw_records[0] if provider.raw_records else {}
    response = raw.get("response", {}) if isinstance(raw, Mapping) else {}
    return {
        "model": model,
        "autosaddler_commit": AUTOSADDLER_COMMIT,
        "diagnosis_model_calls": provider.diagnosis_calls,
        "schema_valid": diagnosis_completed,
        "mutation_finalized": mutation_finalized,
        "mutation_rejection": mutation_rejection,
        "accepted_mutation": accepted,
        "training_score_before": training_before,
        "training_score_after": training_after,
        "training_improvement": (
            training_after - training_before
            if training_before is not None and training_after is not None
            else None
        ),
        "development_score_before": development_before,
        "development_score_after": development_after,
        "development_evaluated": len(development) > 1,
        "development_improvement": (
            development_after - development_before
            if development_before is not None and development_after is not None
            else None
        ),
        "development_regressions": regressions,
        "prompt_tokens": response.get("prompt_eval_count") if isinstance(response, Mapping) else None,
        "generated_tokens": response.get("eval_count") if isinstance(response, Mapping) else None,
        "latency_seconds": (
            round(float(response.get("total_duration", 0)) / 1_000_000_000, 6)
            if isinstance(response, Mapping) and response.get("total_duration") is not None
            else None
        ),
        "done_reason": response.get("done_reason") if isinstance(response, Mapping) else None,
        "provider_error": raw.get("error") if isinstance(raw, Mapping) else None,
        "run_completed": any(event.event_type == "RunCompleted" for event in events),
        "event_count": len(events),
    }


def run_model(model: str) -> dict[str, Any]:
    run_id = f"exp002-v2-{model.replace(':', '-').replace('.', '_')}"
    run_dir = V2_ARTIFACTS / "runs" / model.replace(":", "_").replace(".", "_")
    if run_dir.exists():
        raise RuntimeError(f"refusing to overwrite existing V2 run: {run_dir}")
    provider = HybridOllamaProvider(model)
    engine, store = build_engine(run_dir, run_id, provider)
    engine.run()
    raw_dir = V2_ARTIFACTS / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / f"{model.replace(':', '_')}.json").write_text(
        json.dumps(provider.raw_records, indent=2) + "\n",
        encoding="utf-8",
    )
    return summarize_run(model, store, provider)


def run_comparison() -> int:
    V2_ARTIFACTS.mkdir(parents=True, exist_ok=True)
    summaries = []
    for model in MODELS:
        summaries.append(run_model(model))
    result = {
        "experiment": "002-agent-patch-quality-v2-integration",
        "official_engine": True,
        "autosaddler_commit": AUTOSADDLER_COMMIT,
        "models": list(MODELS),
        "identical_settings": {
            "ollama_options": dict(OLLAMA_OPTIONS),
            "diagnosis_schema": DIAGNOSIS_SCHEMA,
            "train_case_ids": [TRAIN_CASE_ID],
            "development_case_ids": list(DEV_CASE_IDS),
            "iterations": 1,
            "session_retries": 0,
        },
        "results": summaries,
    }
    (V2_ARTIFACTS / "comparison.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


def summarize_existing() -> int:
    summaries = []
    for model in MODELS:
        run_id = f"exp002-v2-{model.replace(':', '-').replace('.', '_')}"
        run_dir = V2_ARTIFACTS / "runs" / model.replace(":", "_").replace(".", "_")
        raw_path = V2_ARTIFACTS / "raw" / f"{model.replace(':', '_')}.json"
        if not run_dir.is_dir() or not raw_path.is_file():
            raise FileNotFoundError(f"completed run artifacts are missing for {model}")
        provider = HybridOllamaProvider(model)
        provider.diagnosis_calls = 1
        loaded = json.loads(raw_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, list):
            raise TypeError(f"raw provider record must be a list: {raw_path}")
        provider.raw_records = loaded
        store = LocalRunStore(run_dir=run_dir, run_id=run_id)
        summaries.append(summarize_run(model, store, provider))
    comparison_path = V2_ARTIFACTS / "comparison.json"
    existing = json.loads(comparison_path.read_text(encoding="utf-8"))
    existing["results"] = summaries
    comparison_path.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(existing, indent=2))
    return 0


def main_check_pin_for_test() -> str:
    completed = subprocess.run(
        ["git", "-C", str(AUTOSADDLER_ROOT), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    actual = completed.stdout.strip()
    if actual != AUTOSADDLER_COMMIT:
        raise RuntimeError(f"AutoSaddler revision mismatch: {actual}")
    status = subprocess.run(
        ["git", "-C", str(AUTOSADDLER_ROOT), "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if status:
        raise RuntimeError("official AutoSaddler checkout has local modifications")
    return actual


def main() -> int:
    parser = argparse.ArgumentParser(description="Experiment 002 AutoSaddler V2 integration")
    parser.add_argument("command", choices=("run", "summarize", "check-pin"))
    args = parser.parse_args()
    if args.command == "check-pin":
        actual = main_check_pin_for_test()
        print(f"AutoSaddler pin verified: {actual}")
        return 0
    if args.command == "summarize":
        return summarize_existing()
    return run_comparison()


if __name__ == "__main__":
    raise SystemExit(main())
