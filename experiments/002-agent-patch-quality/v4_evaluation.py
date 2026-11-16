from __future__ import annotations

import argparse
import asyncio
import copy
import json
import math
import os
import statistics
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from unittest import mock

import v2_integration as v2
import v3_integration as v3

from autosaddler.v2.core.attempts import EventEvaluationAttemptSink
from autosaddler.v2.core.ports import EvaluationContext
from autosaddler.v2.core.run_state import RunState
from autosaddler.v2.core.serde import record


ROOT = Path(__file__).resolve().parent
ARTIFACTS = Path(os.environ.get("EXP002_V4_ARTIFACTS", ROOT / "artifacts" / "v4")).resolve()
MODELS = v2.MODELS
AUTOSADDLER_COMMIT = v2.AUTOSADDLER_COMMIT
PROTOCOL_VERSION = "exp002-v4/1"


def scenario(
    *,
    key: str,
    short: str,
    function: str,
    arguments: tuple[str, ...],
    baseline: str,
    reference: str,
    train: str,
    development: tuple[str, str],
    test: tuple[str, str],
    task: str,
    imports: tuple[str, ...],
    calls: tuple[str, ...],
    diagnosis: tuple[tuple[str, ...], ...],
    cases: str,
) -> tuple[v3.ScenarioDefinition, tuple[str, str]]:
    return (
        v3.ScenarioDefinition(
            key=key,
            short=short,
            function_name=function,
            arguments=arguments,
            baseline=baseline,
            reference=reference,
            train_case=train,
            development_cases=development,
            task=task,
            allowed_imports=imports,
            allowed_calls=calls,
            diagnosis_groups=diagnosis,
            case_block=cases,
        ),
        test,
    )


DEFINITIONS_WITH_TESTS = (
    scenario(
        key="structured_output", short="so", function="extract_result", arguments=("response",),
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
        train="train-json-result-string", development=("dev-result-invalid", "dev-result-copy"),
        test=("test-result-nested", "test-result-malformed"),
        task="Normalize a structured result payload into an independent dictionary and reject unsupported result values.",
        imports=("json",), calls=("dict", "isinstance", "str", "ValueError", "json.loads"),
        diagnosis=(("json string", "json-encoded", "decode", "json.loads"), ("dict", "object")),
        cases='''if case_id == "train-json-result-string":
    assert target({"result": '{"answer": 42}'}) == {"answer": 42}
elif case_id == "dev-result-invalid":
    for value in ('[1, 2]', None):
        try: target({"result": value})
        except ValueError: pass
        else: raise AssertionError("expected ValueError")
elif case_id == "dev-result-copy":
    original = {"answer": 42, "meta": "kept"}; result = target({"result": original})
    assert result == original and result is not original
elif case_id == "test-result-nested":
    assert target({"result": '{"outer": {"value": 7}}'}) == {"outer": {"value": 7}}
elif case_id == "test-result-malformed":
    try: target({"result": "{broken"})
    except ValueError: pass
    else: raise AssertionError("expected ValueError")
else: raise AssertionError("unknown evaluator case")
'''),
    scenario(
        key="tool_arguments", short="ta", function="normalize_arguments", arguments=("arguments",),
        baseline=v2.BASELINE_SOURCE, reference=v2.REFERENCE_SOURCE,
        train="train-json-arguments-string", development=("dev-arguments-invalid", "dev-arguments-copy"),
        test=("test-arguments-empty", "test-arguments-nested"),
        task="Normalize tool-call arguments into an independent dictionary and reject unsupported argument values.",
        imports=("json",), calls=("dict", "isinstance", "str", "ValueError", "json.loads"),
        diagnosis=(("json string", "json-encoded", "decode", "json.loads"), ("dict", "object")),
        cases='''if case_id == "train-json-arguments-string":
    assert target('{"city": "Chicago"}') == {"city": "Chicago"}
elif case_id == "dev-arguments-invalid":
    for value in ('[1, 2]', None):
        try: target(value)
        except ValueError: pass
        else: raise AssertionError("expected ValueError")
elif case_id == "dev-arguments-copy":
    original = {"city": "Chicago", "units": "metric"}; result = target(original)
    assert result == original and result is not original
elif case_id == "test-arguments-empty":
    original = {}; result = target(original); assert result == {} and result is not original
elif case_id == "test-arguments-nested":
    assert target('{"filters": {"active": true}}') == {"filters": {"active": True}}
else: raise AssertionError("unknown evaluator case")
'''),
    scenario(
        key="retries", short="rt", function="should_retry", arguments=("status_code", "attempt", "max_attempts"),
        baseline='''def should_retry(status_code, attempt, max_attempts) -> bool:
    """Return whether another attempt should be made."""
    return status_code >= 500 and attempt <= max_attempts
''',
        reference='''def should_retry(status_code, attempt, max_attempts) -> bool:
    """Return whether another attempt should be made."""
    return status_code in (429, 500, 502, 503, 504) and attempt < max_attempts
''',
        train="train-retry-exhausted", development=("dev-retry-transient", "dev-retry-permanent"),
        test=("test-retry-first-attempt", "test-retry-terminal"),
        task="Decide whether a transient HTTP failure should be retried without exceeding the attempt budget.",
        imports=(), calls=(), diagnosis=(("off-by-one", "boundary", "exhaust", "attempt <", "<= max"),),
        cases='''if case_id == "train-retry-exhausted":
    assert target(503, 3, 3) is False
elif case_id == "dev-retry-transient":
    assert target(503, 2, 3) is True and target(429, 1, 3) is True
elif case_id == "dev-retry-permanent":
    assert target(400, 1, 3) is False and target(503, 3, 3) is False
elif case_id == "test-retry-first-attempt":
    assert target(500, 0, 2) is True
elif case_id == "test-retry-terminal":
    assert target(404, 0, 2) is False and target(504, 2, 2) is False
else: raise AssertionError("unknown evaluator case")
'''),
    scenario(
        key="state_management", short="sm", function="transition_state", arguments=("state", "event"),
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
        train="train-state-immutable", development=("dev-state-fields", "dev-state-independent"),
        test=("test-state-metadata", "test-state-repeat"),
        task="Apply an event to state while preserving the caller's input and all unrelated fields.",
        imports=(), calls=("dict",), diagnosis=(("mutat", "in-place", "same object", "copy"),),
        cases='''if case_id == "train-state-immutable":
    original = {"status": "pending", "count": 1}; result = target(original, "completed")
    assert result == {"status": "completed", "count": 1} and original["status"] == "pending" and result is not original
elif case_id == "dev-state-fields":
    original = {"status": "pending", "owner": "worker-7"}; result = target(original, "running")
    assert result == {"status": "running", "owner": "worker-7"} and original["status"] == "pending"
elif case_id == "dev-state-independent":
    original = {"status": "pending"}; first = target(original, "running"); second = target(original, "failed")
    assert original["status"] == "pending" and first["status"] == "running" and second["status"] == "failed" and first is not second
elif case_id == "test-state-metadata":
    original = {"status": "new", "meta": {"id": 9}}; result = target(original, "queued")
    assert result == {"status": "queued", "meta": {"id": 9}} and original["status"] == "new"
elif case_id == "test-state-repeat":
    original = {"status": "new", "count": 0}; result = target(original, "new")
    assert result == original and result is not original
else: raise AssertionError("unknown evaluator case")
'''),
    scenario(
        key="error_handling", short="eh", function="safe_error_message", arguments=("error",),
        baseline='''def safe_error_message(error) -> str:
    """Return an error message safe for external callers."""
    return str(error)
''',
        reference='''def safe_error_message(error) -> str:
    """Return an error message safe for external callers."""
    return "operation failed"
''',
        train="train-error-token", development=("dev-error-password", "dev-error-empty"),
        test=("test-error-api-key", "test-error-custom"),
        task="Return a stable public error message without exposing underlying exception details.",
        imports=(), calls=("str", "isinstance", "ValueError"),
        diagnosis=(("leak", "secret", "sensitive", "raw error", "str(error)"),),
        cases='''if case_id == "train-error-token":
    result = target(RuntimeError("database failed: token=TRAIN_SECRET")); assert result == "operation failed" and "TRAIN_SECRET" not in result
elif case_id == "dev-error-password":
    result = target(ValueError("password=DEV_PASSWORD")); assert result == "operation failed" and "DEV_PASSWORD" not in result
elif case_id == "dev-error-empty":
    assert target(Exception()) == "operation failed"
elif case_id == "test-error-api-key":
    result = target(RuntimeError("api_key=HELDOUT_KEY")); assert result == "operation failed" and "HELDOUT_KEY" not in result
elif case_id == "test-error-custom":
    class PrivateFailure(Exception): pass
    assert target(PrivateFailure("internal-host-17")) == "operation failed"
else: raise AssertionError("unknown evaluator case")
'''),
    scenario(
        key="pagination", short="pg", function="next_cursor", arguments=("page",),
        baseline='''def next_cursor(page):
    """Return the next usable pagination cursor or None."""
    return page.get("next") or page["cursor"]
''',
        reference='''def next_cursor(page):
    """Return the next usable pagination cursor or None."""
    value = page.get("next")
    if not isinstance(value, str):
        return None
    value = value.strip()
    if not value:
        return None
    return value
''',
        train="train-page-empty", development=("dev-page-whitespace", "dev-page-valid"),
        test=("test-page-missing", "test-page-numeric"),
        task="Return a normalized next-page cursor, or None when no usable cursor is present.",
        imports=(), calls=("isinstance", "str", ".get", ".strip"),
        diagnosis=(("empty", "fallback", "missing", "cursor"),),
        cases='''if case_id == "train-page-empty": assert target({"next": ""}) is None
elif case_id == "dev-page-whitespace": assert target({"next": "   "}) is None
elif case_id == "dev-page-valid": assert target({"next": " abc "}) == "abc"
elif case_id == "test-page-missing": assert target({"items": []}) is None
elif case_id == "test-page-numeric": assert target({"next": 123}) is None
else: raise AssertionError("unknown evaluator case")
'''),
    scenario(
        key="cache_keys", short="ck", function="cache_key", arguments=("namespace", "params"),
        baseline='''def cache_key(namespace, params) -> str:
    """Return a deterministic cache key."""
    return namespace + ":" + str(params)
''',
        reference='''import json

def cache_key(namespace, params) -> str:
    """Return a deterministic cache key."""
    return namespace + ":" + json.dumps(params, sort_keys=True, separators=(",", ":"))
''',
        train="train-cache-order", development=("dev-cache-nested", "dev-cache-namespace"),
        test=("test-cache-empty", "test-cache-boolean"),
        task="Create a deterministic cache key independent of dictionary insertion order.",
        imports=("json",), calls=("str", "json.dumps"),
        diagnosis=(("order", "determin", "canonical", "sort"),),
        cases='''if case_id == "train-cache-order": assert target("n", {"a": 1, "b": 2}) == target("n", {"b": 2, "a": 1})
elif case_id == "dev-cache-nested": assert target("n", {"x": {"a": 1, "b": 2}}) == target("n", {"x": {"b": 2, "a": 1}})
elif case_id == "dev-cache-namespace": assert target("a", {"x": 1}) != target("b", {"x": 1})
elif case_id == "test-cache-empty": assert target("ns", {}) == "ns:{}"
elif case_id == "test-cache-boolean": assert target("ns", {"enabled": True}) == 'ns:{"enabled":true}'
else: raise AssertionError("unknown evaluator case")
'''),
    scenario(
        key="timeout_budget", short="tb", function="effective_timeout", arguments=("requested", "remaining"),
        baseline='''def effective_timeout(requested, remaining):
    """Return a non-negative timeout within the remaining budget."""
    return min(requested, remaining)
''',
        reference='''def effective_timeout(requested, remaining):
    """Return a non-negative timeout within the remaining budget."""
    return max(0, min(requested, remaining))
''',
        train="train-timeout-negative", development=("dev-timeout-cap", "dev-timeout-remaining-negative"),
        test=("test-timeout-zero", "test-timeout-fraction"),
        task="Clamp a requested timeout to the remaining budget without returning a negative duration.",
        imports=(), calls=("min", "max"), diagnosis=(("negative", "clamp", "zero", "non-negative"),),
        cases='''if case_id == "train-timeout-negative": assert target(-2, 10) == 0
elif case_id == "dev-timeout-cap": assert target(30, 8) == 8
elif case_id == "dev-timeout-remaining-negative": assert target(5, -1) == 0
elif case_id == "test-timeout-zero": assert target(0, 0) == 0
elif case_id == "test-timeout-fraction": assert target(2.5, 1.25) == 1.25
else: raise AssertionError("unknown evaluator case")
'''),
    scenario(
        key="event_deduplication", short="ed", function="should_process", arguments=("event_id", "seen_ids"),
        baseline='''def should_process(event_id, seen_ids) -> bool:
    """Return whether an event has not already been seen."""
    return event_id not in seen_ids
''',
        reference='''def should_process(event_id, seen_ids) -> bool:
    """Return whether an event has not already been seen."""
    normalized = event_id.strip()
    normalized = normalized.casefold()
    return normalized not in seen_ids
''',
        train="train-event-normalization", development=("dev-event-case", "dev-event-distinct"),
        test=("test-event-unicode", "test-event-new"),
        task="Detect duplicate event identifiers after whitespace and case normalization.",
        imports=(), calls=(".strip", ".casefold"), diagnosis=(("normaliz", "case", "whitespace", "strip"),),
        cases='''if case_id == "train-event-normalization": assert target(" EVT-1 ", {"evt-1"}) is False
elif case_id == "dev-event-case": assert target("EvT-2", {"evt-2"}) is False
elif case_id == "dev-event-distinct": assert target("evt-3", {"evt-2"}) is True
elif case_id == "test-event-unicode": assert target(" STRASSE ", {"strasse"}) is False
elif case_id == "test-event-new": assert target(" new-id ", set()) is True
else: raise AssertionError("unknown evaluator case")
'''),
    scenario(
        key="permission_scopes", short="ps", function="has_scope", arguments=("granted", "required"),
        baseline='''def has_scope(granted, required) -> bool:
    """Return whether a comma-separated grant contains the exact scope."""
    return required in granted
''',
        reference='''def has_scope(granted, required) -> bool:
    """Return whether a comma-separated grant contains the exact scope."""
    needle = "," + required.strip() + ","
    normalized = "," + granted.replace(" ", "") + ","
    return needle in normalized
''',
        train="train-scope-substring", development=("dev-scope-exact", "dev-scope-prefix"),
        test=("test-scope-whitespace", "test-scope-absent"),
        task="Match an exact permission scope in a comma-separated grant without substring matches.",
        imports=(), calls=(".strip", ".replace"), diagnosis=(("substring", "exact", "delimiter", "boundary"),),
        cases='''if case_id == "train-scope-substring": assert target("read:items:admin", "read:items") is False
elif case_id == "dev-scope-exact": assert target("write:items, read:items", "read:items") is True
elif case_id == "dev-scope-prefix": assert target("read:items-extra", "read:items") is False
elif case_id == "test-scope-whitespace": assert target(" alpha , beta ", "beta") is True
elif case_id == "test-scope-absent": assert target("alpha,beta", "gamma") is False
else: raise AssertionError("unknown evaluator case")
'''),
    scenario(
        key="fallback_selection", short="fs", function="select_endpoint", arguments=("primary", "fallback"),
        baseline='''def select_endpoint(primary, fallback) -> str:
    """Return the first usable endpoint."""
    return primary if primary is not None else fallback
''',
        reference='''def select_endpoint(primary, fallback) -> str:
    """Return the first usable endpoint."""
    if isinstance(primary, str) and primary.strip():
        return primary.strip()
    if isinstance(fallback, str) and fallback.strip():
        return fallback.strip()
    raise ValueError("no usable endpoint")
''',
        train="train-fallback-blank-primary", development=("dev-fallback-primary", "dev-fallback-none"),
        test=("test-fallback-null-primary", "test-fallback-blank-both"),
        task="Select the first nonblank endpoint and reject requests with no usable endpoint.",
        imports=(), calls=("isinstance", "str", "ValueError", ".strip"),
        diagnosis=(("blank", "empty", "whitespace", "fallback"),),
        cases='''if case_id == "train-fallback-blank-primary": assert target("   ", "backup") == "backup"
elif case_id == "dev-fallback-primary": assert target(" main ", "backup") == "main"
elif case_id == "dev-fallback-none":
    try: target(None, None)
    except ValueError: pass
    else: raise AssertionError("expected ValueError")
elif case_id == "test-fallback-null-primary": assert target(None, " reserve ") == "reserve"
elif case_id == "test-fallback-blank-both":
    try: target(" ", "  ")
    except ValueError: pass
    else: raise AssertionError("expected ValueError")
else: raise AssertionError("unknown evaluator case")
'''),
    scenario(
        key="rate_limit_delay", short="rl", function="retry_delay", arguments=("retry_after", "default", "maximum"),
        baseline='''def retry_delay(retry_after, default, maximum):
    """Return a bounded non-negative retry delay."""
    return min(float(retry_after), maximum)
''',
        reference='''def retry_delay(retry_after, default, maximum):
    """Return a bounded non-negative retry delay."""
    if retry_after is None:
        value = default
    elif isinstance(retry_after, (int, float)):
        value = float(retry_after)
    elif isinstance(retry_after, str) and retry_after.strip().isdigit():
        value = float(retry_after)
    else:
        value = default
    return max(0, min(value, maximum))
''',
        train="train-delay-missing", development=("dev-delay-header", "dev-delay-invalid"),
        test=("test-delay-negative", "test-delay-maximum"),
        task="Parse a retry delay with a default and clamp it to a non-negative maximum.",
        imports=(), calls=("min", "max", "float", "isinstance", "int", "str", ".strip", ".isdigit"),
        diagnosis=(("none", "missing", "default", "float"),),
        cases='''if case_id == "train-delay-missing": assert target(None, 4, 30) == 4
elif case_id == "dev-delay-header": assert target("12", 4, 30) == 12
elif case_id == "dev-delay-invalid": assert target("later", 4, 30) == 4
elif case_id == "test-delay-negative": assert target(-5, 4, 30) == 0
elif case_id == "test-delay-maximum": assert target(90, 4, 30) == 30
else: raise AssertionError("unknown evaluator case")
'''),
    scenario(
        key="completion_tracking", short="ct", function="is_complete", arguments=("required", "completed"),
        baseline='''def is_complete(required, completed) -> bool:
    """Return whether every required item is complete."""
    return bool(completed)
''',
        reference='''def is_complete(required, completed) -> bool:
    """Return whether every required item is complete."""
    return required.issubset(completed)
''',
        train="train-completion-partial", development=("dev-completion-all", "dev-completion-empty"),
        test=("test-completion-extra", "test-completion-missing"),
        task="Report completion only when every required item is present in the completed set.",
        imports=(), calls=("bool", ".issubset"), diagnosis=(("all", "subset", "partial", "required"),),
        cases='''if case_id == "train-completion-partial": assert target({"a", "b"}, {"a"}) is False
elif case_id == "dev-completion-all": assert target({"a", "b"}, {"a", "b"}) is True
elif case_id == "dev-completion-empty": assert target(set(), set()) is True
elif case_id == "test-completion-extra": assert target({"a"}, {"a", "b"}) is True
elif case_id == "test-completion-missing": assert target({"a", "b", "c"}, {"a", "c"}) is False
else: raise AssertionError("unknown evaluator case")
'''),
    scenario(
        key="idempotency_headers", short="ih", function="idempotency_key", arguments=("headers",),
        baseline='''def idempotency_key(headers) -> str:
    """Return a normalized idempotency key."""
    return headers.get("Idempotency-Key")
''',
        reference='''def idempotency_key(headers) -> str:
    """Return a normalized idempotency key."""
    value = headers.get("Idempotency-Key")
    if value is None:
        value = headers.get("idempotency-key")
    if value is None:
        value = headers.get("IDEMPOTENCY-KEY")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("missing idempotency key")
    return value.strip()
''',
        train="train-header-lowercase", development=("dev-header-trim", "dev-header-missing"),
        test=("test-header-uppercase", "test-header-nonstring"),
        task="Read a case-insensitive idempotency header, trim it, and reject missing or invalid keys.",
        imports=(), calls=("isinstance", "str", "ValueError", ".get", ".strip"),
        diagnosis=(("case", "lower", "header", "idempotency"),),
        cases='''if case_id == "train-header-lowercase": assert target({"idempotency-key": " key-1 "}) == "key-1"
elif case_id == "dev-header-trim": assert target({"Idempotency-Key": " abc "}) == "abc"
elif case_id == "dev-header-missing":
    try: target({})
    except ValueError: pass
    else: raise AssertionError("expected ValueError")
elif case_id == "test-header-uppercase": assert target({"IDEMPOTENCY-KEY": "XYZ"}) == "XYZ"
elif case_id == "test-header-nonstring":
    try: target({"Idempotency-Key": 17})
    except ValueError: pass
    else: raise AssertionError("expected ValueError")
else: raise AssertionError("unknown evaluator case")
'''),
    scenario(
        key="status_normalization", short="sn", function="normalize_status", arguments=("status",),
        baseline='''def normalize_status(status) -> str:
    """Return a canonical success or failure status."""
    return status.lower()
''',
        reference='''def normalize_status(status) -> str:
    """Return a canonical success or failure status."""
    normalized = status.strip()
    normalized = normalized.casefold()
    if normalized in ("ok", "success", "succeeded"):
        return "success"
    if normalized in ("error", "failed", "failure"):
        return "failure"
    raise ValueError("unknown status")
''',
        train="train-status-ok", development=("dev-status-failed", "dev-status-unknown"),
        test=("test-status-success", "test-status-failure"),
        task="Normalize common status aliases into success or failure and reject unknown values.",
        imports=(), calls=("ValueError", ".lower", ".strip", ".casefold"),
        diagnosis=(("strip", "whitespace", "alias", "canonical", "normaliz"),),
        cases='''if case_id == "train-status-ok": assert target(" OK ") == "success"
elif case_id == "dev-status-failed": assert target("Failed") == "failure"
elif case_id == "dev-status-unknown":
    try: target("pending")
    except ValueError: pass
    else: raise AssertionError("expected ValueError")
elif case_id == "test-status-success": assert target("sUcCeSs") == "success"
elif case_id == "test-status-failure": assert target(" FAILURE ") == "failure"
else: raise AssertionError("unknown evaluator case")
'''),
)

SCENARIOS = tuple(item[0] for item in DEFINITIONS_WITH_TESTS)
TEST_CASES = {item[0].key: item[1] for item in DEFINITIONS_WITH_TESTS}
SCENARIO_BY_KEY = {item.key: item for item in SCENARIOS}
PREFLIGHT_KEYS = tuple(item.key for item in SCENARIOS[:5])


def build_engine(definition: v3.ScenarioDefinition, run_dir: Path, run_id: str, provider):
    store = v2.ProjectionRetryStore(run_dir=run_dir, run_id=run_id)
    components = v3.build_scenario(definition, run_dir, store)
    config: Mapping[str, v2.JsonValue] = {
        "schema_version": "autosaddler/v2",
        "protocol": PROTOCOL_VERSION,
        "scenario": {"type": definition.key, "version": "1"},
        "optimization": {"max_iterations": 1, "max_rollouts": 10, "session_retries": 0},
        "provider": {"type": "ollama-exp002", "model": provider.model, "options": dict(v2.OLLAMA_OPTIONS)},
    }
    store.initialize(resolved_config=config, resolved_entities=components.resolved_entities)
    policies = v2.PolicyBundle(
        task_selection=v2.FixedTaskSelectionPolicy(batch_size=1),
        acceptance=v2.MatchedValidStrictImprovement(),
        development=v2.FullOnAcceptDevelopment(),
        ranking=v2.MeanDevelopmentRanking(),
        budget=v2.BudgetPolicy(max_rollouts=10, max_iterations=1),
    )
    engine = v2.AutoSaddlerEngine(
        store=store, scenario=components, provider=provider, policies=policies,
        diagnosis_patch_timeout_seconds=360, selection_timeout_seconds=30,
        reflection_timeout_seconds=30, session_retries=0,
    )
    return engine, store, components


async def _evaluate_test_candidate(store, components, candidate, cases, operation_id):
    store.append("EvaluationStarted", operation_id, {
        "candidate_id": candidate.candidate_id, "case_ids": [case.case_id for case in cases],
        "split": "test", "purpose": "test", "iteration": None,
    })
    evaluation_id = v2.sha256_digest(operation_id)
    context = EvaluationContext(
        run_id=store.run_id, operation_id=operation_id, iteration=None, purpose="test", split="test",
        repetitions=1, capture_traces=False,
        artifact_dir=store.run_dir / "post_optimization" / "test" / evaluation_id.removeprefix("sha256:"),
        attempt_sink=EventEvaluationAttemptSink(store=store, evaluation_operation_id=operation_id),
    )
    evaluation = await components.evaluator.evaluate(candidate, cases, context)
    ref = store.write_json(
        f"post_optimization/test/{evaluation.evaluation_id.removeprefix('sha256:')}/evaluation.json",
        record(evaluation), kind="evaluation",
    )
    store.append("EvaluationCompleted", operation_id, {"evaluation": record(evaluation), "artifact": record(ref)})
    return evaluation


def evaluate_heldout(store, components, optimization_result, definition):
    state = RunState.replay(store.events())
    run_started = store.events_of_type("RunStarted")[0]
    seed_id = str(run_started.payload["seed_candidate"]["candidate_id"])
    seed = state.candidates[seed_id]
    selected = state.candidates[optimization_result.selected_candidate_id]
    cases = tuple(v2.Case(case_id=case_id, split="test", payload={}) for case_id in TEST_CASES[definition.key])
    baseline = asyncio.run(_evaluate_test_candidate(store, components, seed, cases, f"{store.run_id}:heldout:baseline"))
    final = asyncio.run(_evaluate_test_candidate(store, components, selected, cases, f"{store.run_id}:heldout:selected"))
    before = {observation.case_id: observation.score for observation in baseline.observations}
    after = {observation.case_id: observation.score for observation in final.observations}
    regressions = [case_id for case_id, score in before.items() if score is not None and score > (after.get(case_id) or 0)]
    return {
        "selected_candidate_id": selected.candidate_id,
        "seed_candidate_id": seed.candidate_id,
        "heldout_score_before": baseline.aggregate_score,
        "heldout_score_after": final.aggregate_score,
        "heldout_improvement": final.aggregate_score - baseline.aggregate_score,
        "heldout_full_success": final.aggregate_score == 1.0,
        "heldout_regressions": regressions,
    }


def model_slug(model: str) -> str:
    return "q17" if model == "qwen3:1.7b" else "q4"


def run_one(definition, model):
    run_id = f"exp002-v4-{definition.key}-{model_slug(model)}"
    run_dir = ARTIFACTS / "runs" / definition.short / model_slug(model)
    raw_path = ARTIFACTS / "raw" / f"{definition.short}-{model_slug(model)}.json"
    if run_dir.exists() or raw_path.exists():
        raise RuntimeError(f"refusing to repeat V4 model call: {definition.key}/{model}")
    provider = v2.HybridOllamaProvider(model)
    engine, store, components = build_engine(definition, run_dir, run_id, provider)
    optimization = engine.run()
    ARTIFACTS.joinpath("raw").mkdir(parents=True, exist_ok=True)
    raw_path.write_text(json.dumps(provider.raw_records, indent=2) + "\n", encoding="utf-8")
    summary = v3.summarize_run(definition, model, store, provider)
    summary.update(evaluate_heldout(store, components, optimization, definition))
    summary["protocol"] = PROTOCOL_VERSION
    with (ARTIFACTS / "ledger.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(v2.canonical_json(summary) + "\n")
    return summary


def rows():
    path = ARTIFACTS / "ledger.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


def normalized_request(record):
    request = copy.deepcopy(record["request"]); request["model"] = "MODEL"; return request


def verify_pair(definition):
    records = []
    for model in MODELS:
        path = ARTIFACTS / "raw" / f"{definition.short}-{model_slug(model)}.json"
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if len(loaded) != 1 or loaded[0].get("error"):
            raise RuntimeError(f"provider interaction failure: {definition.key}/{model}")
        records.append(loaded[0])
    if v2.canonical_json(normalized_request(records[0])) != v2.canonical_json(normalized_request(records[1])):
        raise RuntimeError(f"non-identical model-facing evidence: {definition.key}")


def protocol_document():
    return {
        "schema_version": "exp002-v4-preregistration/v1", "protocol": PROTOCOL_VERSION,
        "official_engine_commit": AUTOSADDLER_COMMIT, "models": list(MODELS),
        "estimated_model_calls": 30, "trials_per_model_scenario": 1,
        "settings": {**dict(v2.OLLAMA_OPTIONS), "think": False, "model_retries": 0},
        "schema": v2.DIAGNOSIS_SCHEMA,
        "acceptance": "official MatchedValidStrictImprovement",
        "development_gate": "official FullOnAcceptDevelopment",
        "ranking": "official MeanDevelopmentRanking",
        "storage_projection_retry": {"exceptions": ["PermissionError"], "attempts": 8, "delays_seconds": [0.025 * i for i in range(1, 8)]},
        "preflight_gate": {
            "scenario_count": 5, "required_calls": 10,
            "requirements": ["one raw call per cell", "no provider error", "byte-identical normalized requests", "completed engine run", "two post-selection heldout evaluations"],
            "model-quality_failures_are_outcomes_not_integration_failures": True,
        },
        "metrics": {
            "diagnosis_accuracy": "all preregistered concept groups contain at least one case-insensitive substring",
            "schema_validity": "official SessionCompleted for diagnose_patch",
            "mutation_validity": "official CandidateFinalized",
            "training_acceptance": "official acceptance verdict",
            "development_improvement": "accepted-child development aggregate minus seed development aggregate",
            "heldout_improvement": "post-selection test aggregate minus seed test aggregate",
            "heldout_success": "post-selection test aggregate equals 1.0",
            "regression": "any previously passing case scores lower",
            "cost": "Ollama prompt_eval_count, eval_count, total_duration",
            "confidence_intervals": "two-sided Wilson score interval, 95%, scenario as independent unit",
            "latency_p95": "linear interpolation at rank 0.95*(n-1)",
            "efficiency": "count of positive heldout improvements per total Ollama inference minute",
        },
        "heldout_isolation": "test cases absent from ScenarioComponents, prompts, evidence, development ranking, and resolved entities; evaluated only after RunCompleted",
        "scenarios": [{
            "key": item.key, "task": item.task, "function": item.function_name,
            "train": [item.train_case], "development": list(item.development_cases), "test": list(TEST_CASES[item.key]),
            "baseline_sha256": v2.sha256_digest(item.baseline), "reference_sha256": v2.sha256_digest(item.reference),
            "case_runner_sha256": v2.sha256_digest(v3.runner_source(item)),
            "allowed_imports": list(item.allowed_imports), "allowed_calls": list(item.allowed_calls),
            "diagnosis_groups": [list(group) for group in item.diagnosis_groups],
        } for item in SCENARIOS],
    }


def preregister():
    if ARTIFACTS.exists():
        raise RuntimeError("refusing to overwrite V4 preregistration or results")
    ARTIFACTS.mkdir(parents=True)
    payload = v2.canonical_json(protocol_document()) + "\n"
    (ARTIFACTS / "preregistration.json").write_text(payload, encoding="utf-8")
    digest = v2.sha256_digest(payload.rstrip("\n"))
    (ARTIFACTS / "preregistration.sha256").write_text(digest + "\n", encoding="utf-8")
    print(digest)


def verify_protocol():
    expected = v2.canonical_json(protocol_document()) + "\n"
    actual = (ARTIFACTS / "preregistration.json").read_text(encoding="utf-8")
    digest = v2.sha256_digest(actual.rstrip("\n"))
    if actual != expected or (ARTIFACTS / "preregistration.sha256").read_text().strip() != digest:
        raise RuntimeError("frozen V4 protocol mismatch")
    return digest


def run_batch(definitions, expected_existing):
    verify_protocol()
    if len(rows()) != expected_existing:
        raise RuntimeError(f"expected {expected_existing} existing rows")
    for definition in definitions:
        for model in MODELS:
            result = run_one(definition, model)
            if result.get("provider_error") or not result.get("run_completed"):
                raise RuntimeError(f"major integration failure: {definition.key}/{model}")
        verify_pair(definition)


def gate_preflight():
    current = rows()
    if len(current) != 10 or {row["scenario"] for row in current} != set(PREFLIGHT_KEYS):
        raise RuntimeError("preflight does not contain the frozen ten cells")
    for definition in SCENARIOS[:5]: verify_pair(definition)
    gate = {"valid": True, "rows": 10, "scenarios": list(PREFLIGHT_KEYS), "protocol_sha256": verify_protocol()}
    (ARTIFACTS / "preflight_gate.json").write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(gate, indent=2))


def wilson(successes, n, z=1.959963984540054):
    if n == 0: return [None, None]
    p = successes / n; denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return [center - margin, center + margin]


def percentile95(values):
    ordered = sorted(values); position = 0.95 * (len(ordered) - 1); low = math.floor(position); high = math.ceil(position)
    return ordered[low] if low == high else ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def aggregate(all_rows):
    output = {}
    for model in MODELS:
        selected = [row for row in all_rows if row["model"] == model]; n = len(selected)
        counts = {
            "diagnosis_accurate": sum(bool(row["diagnosis_accurate"]) for row in selected),
            "schema_valid": sum(bool(row["schema_valid"]) for row in selected),
            "mutation_valid": sum(bool(row["mutation_finalized"]) for row in selected),
            "training_accepted": sum(bool(row["accepted_mutation"]) for row in selected),
            "development_improved": sum((row["development_improvement"] or 0) > 0 for row in selected),
            "development_regression": sum(bool(row["development_regressions"]) for row in selected),
            "heldout_improved": sum(row["heldout_improvement"] > 0 for row in selected),
            "heldout_full_success": sum(bool(row["heldout_full_success"]) for row in selected),
            "heldout_regression": sum(bool(row["heldout_regressions"]) for row in selected),
        }
        latencies = [float(row["latency_seconds"]) for row in selected]
        total_latency = sum(latencies)
        output[model] = {
            "n": n, "counts": counts,
            "rates_with_wilson95": {key: {"rate": value / n, "ci": wilson(value, n)} for key, value in counts.items()},
            "prompt_tokens": sum(row["prompt_tokens"] for row in selected),
            "generated_tokens": sum(row["generated_tokens"] for row in selected),
            "latency_seconds_total": total_latency, "latency_seconds_mean": statistics.mean(latencies),
            "latency_seconds_p95": percentile95(latencies),
            "heldout_improvements_per_inference_minute": counts["heldout_improved"] / (total_latency / 60),
        }
    return output


def audit():
    verify_protocol(); all_rows = rows()
    if len(all_rows) != 30: raise RuntimeError("expected 30 result rows")
    checks = []
    for definition in SCENARIOS:
        verify_pair(definition)
        pair = [row for row in all_rows if row["scenario"] == definition.key]
        if len(pair) != 2: raise RuntimeError(f"missing pair: {definition.key}")
        for row in pair:
            raw_records = json.loads((ARTIFACTS / "raw" / f"{definition.short}-{model_slug(row['model'])}.json").read_text())
            raw = raw_records[0]
            response = raw["response"]
            if row["prompt_tokens"] != response.get("prompt_eval_count") or row["generated_tokens"] != response.get("eval_count"):
                raise RuntimeError(f"token mismatch: {definition.key}/{row['model']}")
            if row["diagnosis_model_calls"] != 1: raise RuntimeError("model call count mismatch")
            run_id = f"exp002-v4-{definition.key}-{model_slug(row['model'])}"
            run_dir = ARTIFACTS / "runs" / definition.short / model_slug(row["model"])
            store = v2.ProjectionRetryStore(run_dir=run_dir, run_id=run_id)
            provider = v2.HybridOllamaProvider(row["model"])
            provider.diagnosis_calls = 1
            provider.raw_records = raw_records
            recomputed = v3.summarize_run(definition, row["model"], store, provider)
            for field in (
                "schema_valid", "mutation_finalized", "mutation_rejection", "accepted_mutation",
                "training_score_before", "training_score_after", "training_improvement",
                "development_score_before", "development_score_after", "development_improvement",
                "development_regressions", "prompt_tokens", "generated_tokens", "latency_seconds",
            ):
                if row[field] != recomputed[field]:
                    raise RuntimeError(f"event audit mismatch: {definition.key}/{row['model']}/{field}")
            events = store.events()
            test_evaluations = {
                event.operation_id: v2.evaluation_from(event.payload.get("evaluation"))
                for event in events if event.event_type == "EvaluationCompleted" and ":heldout:" in event.operation_id
            }
            baseline = test_evaluations.get(f"{run_id}:heldout:baseline")
            final = test_evaluations.get(f"{run_id}:heldout:selected")
            if baseline is None or final is None:
                raise RuntimeError(f"missing heldout evaluation: {definition.key}/{row['model']}")
            if row["heldout_score_before"] != baseline.aggregate_score or row["heldout_score_after"] != final.aggregate_score:
                raise RuntimeError(f"heldout audit mismatch: {definition.key}/{row['model']}")
            selected_events = [event for event in events if event.event_type == "BestCandidateSelected"]
            if len(selected_events) != 1 or row["selected_candidate_id"] != selected_events[0].payload.get("candidate_id"):
                raise RuntimeError(f"selection audit mismatch: {definition.key}/{row['model']}")
        checks.append({"scenario": definition.key, "normalized_requests_identical": True})
    summary = {"schema_version": "exp002-v4-audit/v1", "protocol_sha256": verify_protocol(), "rows": 30, "checks": checks, "aggregate": aggregate(all_rows)}
    (ARTIFACTS / "results.json").write_text(json.dumps({"protocol_sha256": verify_protocol(), "results": all_rows, "aggregate": summary["aggregate"]}, indent=2) + "\n", encoding="utf-8")
    (ARTIFACTS / "audit.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary["aggregate"], indent=2))


def write_svg(path, title, labels, series):
    width, height = 900, 460; margin = 70; chart_h = 300; group_w = (width - 2 * margin) / len(labels); colors = ["#2563eb", "#f97316"]
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="white"/>', f'<text x="{width/2}" y="32" text-anchor="middle" font-family="sans-serif" font-size="20">{title}</text>']
    for tick in range(6):
        y = margin + chart_h - tick * chart_h / 5; parts.append(f'<line x1="{margin}" y1="{y}" x2="{width-margin}" y2="{y}" stroke="#ddd"/>'); parts.append(f'<text x="{margin-8}" y="{y+4}" text-anchor="end" font-family="sans-serif" font-size="11">{tick*20}%</text>')
    for i, label in enumerate(labels):
        x0 = margin + i * group_w
        for j, values in enumerate(series):
            value = values[1][i]; bar_w = group_w * 0.3; x = x0 + group_w * (0.17 + j * 0.34); bar_h = value * chart_h; y = margin + chart_h - bar_h
            parts.append(f'<rect x="{x}" y="{y}" width="{bar_w}" height="{bar_h}" fill="{colors[j]}"/>'); parts.append(f'<text x="{x+bar_w/2}" y="{y-5}" text-anchor="middle" font-family="sans-serif" font-size="10">{value*100:.0f}%</text>')
        parts.append(f'<text x="{x0+group_w/2}" y="{margin+chart_h+22}" text-anchor="middle" font-family="sans-serif" font-size="11">{label}</text>')
    for j, values in enumerate(series): parts.append(f'<rect x="{margin+j*180}" y="{height-35}" width="14" height="14" fill="{colors[j]}"/><text x="{margin+20+j*180}" y="{height-23}" font-family="sans-serif" font-size="12">{values[0]}</text>')
    parts.append('</svg>'); path.write_text("\n".join(parts), encoding="utf-8")


def charts():
    data = json.loads((ARTIFACTS / "audit.json").read_text())["aggregate"]; chart_dir = ARTIFACTS / "charts"; chart_dir.mkdir(exist_ok=True)
    keys = ["schema_valid", "mutation_valid", "training_accepted", "development_improved", "heldout_improved", "heldout_full_success"]
    labels = ["Schema", "Mutation", "Train accept", "Dev improve", "Test improve", "Test full"]
    series = [(model, [data[model]["rates_with_wilson95"][key]["rate"] for key in keys]) for model in MODELS]
    write_svg(chart_dir / "quality_rates.svg", "AutoSaddler V2 local-model outcome rates", labels, series)
    width, height = 760, 430; margin = 80; chart_h = 270; colors = ["#2563eb", "#f97316"]
    values = [(data[model]["latency_seconds_mean"], data[model]["latency_seconds_p95"]) for model in MODELS]
    maximum = max(item for pair in values for item in pair) * 1.15
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="white"/>', f'<text x="{width/2}" y="32" text-anchor="middle" font-family="sans-serif" font-size="20">Local inference latency</text>']
    for tick in range(6):
        seconds = maximum * tick / 5; y = margin + chart_h - chart_h * tick / 5
        parts.append(f'<line x1="{margin}" y1="{y}" x2="{width-margin}" y2="{y}" stroke="#ddd"/><text x="{margin-8}" y="{y+4}" text-anchor="end" font-family="sans-serif" font-size="11">{seconds:.0f}s</text>')
    for i, model in enumerate(MODELS):
        center = margin + (i + 0.5) * (width - 2 * margin) / 2
        for j, (name, value) in enumerate(zip(("mean", "p95"), values[i])):
            bar_w = 70; x = center - 85 + j * 100; bar_h = value / maximum * chart_h; y = margin + chart_h - bar_h
            parts.append(f'<rect x="{x}" y="{y}" width="{bar_w}" height="{bar_h}" fill="{colors[j]}"/><text x="{x+bar_w/2}" y="{y-6}" text-anchor="middle" font-family="sans-serif" font-size="11">{value:.1f}s</text>')
        parts.append(f'<text x="{center}" y="{margin+chart_h+25}" text-anchor="middle" font-family="sans-serif" font-size="13">{model}</text>')
    parts.append(f'<rect x="{margin}" y="{height-38}" width="14" height="14" fill="{colors[0]}"/><text x="{margin+20}" y="{height-26}" font-family="sans-serif" font-size="12">mean</text><rect x="{margin+100}" y="{height-38}" width="14" height="14" fill="{colors[1]}"/><text x="{margin+120}" y="{height-26}" font-family="sans-serif" font-size="12">p95</text></svg>')
    (chart_dir / "latency.svg").write_text("\n".join(parts), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("command", choices=("preregister", "run-preflight", "gate-preflight", "run-remaining", "audit", "charts", "check-pin")); args = parser.parse_args()
    if args.command == "preregister": preregister()
    elif args.command == "run-preflight": run_batch(SCENARIOS[:5], 0)
    elif args.command == "gate-preflight": gate_preflight()
    elif args.command == "run-remaining":
        gate = json.loads((ARTIFACTS / "preflight_gate.json").read_text());
        if gate.get("valid") is not True: raise RuntimeError("preflight gate is not valid")
        run_batch(SCENARIOS[5:], 10)
    elif args.command == "audit": audit()
    elif args.command == "charts": charts()
    else: print(v2.main_check_pin_for_test())


if __name__ == "__main__": main()
