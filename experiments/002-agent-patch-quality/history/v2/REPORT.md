# AutoSaddler V2 local-provider comparison

## Scope and setup

This is one deterministic custom scenario, not a benchmark. It uses the official AutoSaddler V2 engine at commit `e841543ce6cf6bdf690e65134b12a829ec9f66d1`, `ComponentMapHarnessSpace`, evidence/session lifecycle, matched-valid strict-improvement acceptance, full-on-accept development gating, and mean development ranking. The official checkout remained unmodified. Only the Ollama provider, prompt pack, evaluator, and scenario adapter are experiment code.

The run used Ollama 0.40.16 on an Intel Core 5 120U with 12 logical CPUs, 15.72 GiB RAM, and Intel integrated graphics (CPU inference). Both models were Q4_K_M: `qwen3:1.7b` (`8f68893c685c`) and `qwen3:4b` (`359d7dd4bcda`). Each received one native JSON-schema-constrained diagnosis call with temperature 0, seed 42, `num_ctx=4096`, `num_predict=1200`, thinking disabled, and no retries. Evolution and reflection were deterministic controls.

## Results

| Model | Schema valid | Mutation finalized | Accepted | Train before -> after | Development after | Regressions | Prompt / generated tokens | Ollama latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3:1.7b | yes | yes | no | 0.0 -> 0.0 | not gated in | not measured | 590 / 176 | 16.134 s |
| qwen3:4b | yes | no | no | 0.0 -> not evaluated | not gated in | not measured | 588 / 274 | 49.756 s |

Valid structured responses: 2/2. Finalized mutations: 1/2. Accepted mutations and training improvements: 0/2. Development improvements and regressions cannot be reported because neither candidate passed the official training acceptance gate.

## Failures and interpretation

- The 1.7B model misdiagnosed a JSON object string as requiring a list/tuple. Its syntactically valid component update finalized, failed the training case, and was declined by `MatchedValidStrictImprovement`.
- The 4B model identified the JSON-decoding cause, but its update called `type()`, which the disclosed scenario verifier did not permit. V2 rejected it before training evaluation. Inspection also shows the proposed code returns dictionary inputs without copying and does not require decoded JSON to be an object; these are likely development issues, but they were not evaluated and are not counted results.
- Candidate execution used a stripped environment, an isolated per-case directory, socket blocking, a 15-second timeout, and an AST allowlist. This is defense in depth, not a hardened OS/container sandbox.
- The completed calls received semantically identical evidence and identical instructions/settings, but the raw tracebacks contained different random temporary-directory paths. Therefore the literal prompts were not byte-identical. No model was rerun because the protocol allowed only one attempt. The evaluator now replaces that volatile path with `<isolated-workspace>`, and a regression test verifies stable traces for future runs.

## Verification and conclusion

The integration test command was `.venv\Scripts\python.exe -m unittest discover -s tests -v`; all 12 tests passed after the run. The pin check was `.venv\Scripts\python.exe v2_integration.py check-pin`. Raw Ollama requests/responses, event logs, evaluations, and the machine-readable comparison are under `artifacts/v2/`.

The integration faithfully exercises the official V2 optimization workflow mechanics and structured component-mutation path. It does not reproduce the paper's task agents, benchmark workloads, Claude Agent SDK behavior, or reported performance, and the literal-prompt defect weakens this two-model comparison.

A small expansion is worthwhile only as a new, preregistered run after retaining the trace sanitizer and adding multiple deterministic scenarios. These results alone provide no evidence that either tested local model can reliably fill AutoSaddler's diagnosis-and-modification role.
