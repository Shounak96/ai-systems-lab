# Experiment 002 — AutoSaddler V2 Local-Model Harness Optimization

Can inexpensive local Qwen models diagnose agent-harness failures and generate improvements that generalize beyond the observed training failure—and what does that capability cost in local inference time?

This experiment adapts the diagnosis and structured component-mutation role from [Microsoft AutoSaddler](https://github.com/microsoft/AutoSaddler) to Ollama. It uses AutoSaddler V2's official optimization engine, matched training acceptance, development gate, and ranking without modifying upstream core logic. The pinned upstream revision is [`e841543ce6cf6bdf690e65134b12a829ec9f66d1`](AUTOSADDLER_PIN.txt).

## Main result: V4

V4 preregistered 15 independent harness failures. Every scenario had one training case, two development cases, and two final held-out cases. Held-out cases were excluded from optimizer prompts, evidence, acceptance, development ranking, and resolved engine entities; they were evaluated only after optimization against the engine-selected harness.

Both models produced schema-valid responses in all 15 scenarios:

| Metric | Qwen3 1.7B Q4_K_M | Qwen3 4B Q4_K_M |
|---|---:|---:|
| Valid mutations | 10/15 | 12/15 |
| Training accepted | 7/15 | 12/15 |
| Development improved | 4/15 | 8/15 |
| Selected harness improved held-out score | 5/15 (33.3%, Wilson 95% CI 15.2–58.3%) | 8/15 (53.3%, Wilson 95% CI 30.1–75.2%) |
| Mean / p95 local latency | 15.33 / 18.34 s | 38.07 / 42.32 s |
| Held-out improvements per inference minute | 1.304 | 0.841 |

The 4B model produced more absolute generalizing improvements and used fewer tokens per improvement. The 1.7B model was 2.48× faster on the test CPU and produced more improvements per inference minute. Confidence intervals are wide and overlap; this is descriptive evidence from a synthetic, single-trial study—not a statistical-superiority or AutoSaddler-paper-equivalence claim.

[Technical report](V4_TECHNICAL_REPORT.md) · [Preregistration](artifacts/v4/preregistration.json) · [Audited results](artifacts/v4/results.json) · [Raw model records](artifacts/v4/raw/) · [Quality chart](artifacts/v4/charts/quality_rates.svg) · [Latency chart](artifacts/v4/charts/latency.svg)

## Implementation and methodology

- `v2_integration.py`: pinned AutoSaddler imports, Ollama provider, structured-output contract, isolated evaluator, and deterministic control stages.
- `v3_integration.py`: reusable multi-scenario adapter and model-facing prompt construction.
- `v4_evaluation.py`: 15 frozen scenario definitions, train/development/held-out orchestration, preflight gate, Wilson intervals, raw-event audit, and chart generation.
- `tests/`: baseline/reference solvability, hidden-test isolation, byte-identical normalized prompts, filesystem/network restrictions, official workflow integration, and storage retry tests.
- `artifacts/v4/runs/*/*/events.jsonl`: curated official-engine event streams used by the shipped audit. Generated candidates, sessions, workspaces, and duplicate projections are intentionally excluded.

Candidate code executes in disposable directories with a stripped environment, blocked socket creation, an AST operation allowlist, and timeouts. This is defense in depth, not a hardened container or separate operating-system security boundary.

The Windows `LocalRunStore` workaround subclasses the official store and retries only transient `PermissionError` failures while refreshing derived projection files. It never retries model calls or changes optimization decisions.

## Setup

Requirements: Python 3.12, Git, Ollama 0.40.16 or compatible, and locally installed `qwen3:1.7b` and `qwen3:4b`. Model weights and the upstream repository are not bundled.

```powershell
cd experiments/002-agent-patch-quality
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-v2.txt

git clone https://github.com/microsoft/AutoSaddler.git vendor/AutoSaddler
git -C vendor/AutoSaddler checkout e841543ce6cf6bdf690e65134b12a829ec9f66d1
.\.venv\Scripts\python.exe v4_evaluation.py check-pin
```

Only `jsonschema` and `PyYAML` are direct Python dependencies for this isolated integration. Ollama models can be verified with `ollama list`.

## Verify published artifacts

These commands do not invoke the models:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe v4_evaluation.py audit
.\.venv\Scripts\python.exe v4_evaluation.py charts
```

The audit reconciles 30 result rows with raw Ollama counters, official session/evaluation/acceptance events, candidate selection, and held-out evaluations.

## Reproduce V4 from scratch

Use a separate ignored output directory so the published artifacts remain unchanged:

```powershell
$env:EXP002_V4_ARTIFACTS = (Join-Path $PWD "reproduction/v4")
.\.venv\Scripts\python.exe v4_evaluation.py preregister
.\.venv\Scripts\python.exe v4_evaluation.py run-preflight
.\.venv\Scripts\python.exe v4_evaluation.py gate-preflight
.\.venv\Scripts\python.exe v4_evaluation.py run-remaining
.\.venv\Scripts\python.exe v4_evaluation.py audit
.\.venv\Scripts\python.exe v4_evaluation.py charts
```

The published study used one call per model/scenario: 30 sequential Ollama calls and no model retries or post-generation repair. CPU inference took about 13.35 minutes in aggregate.

## Experiment history

Earlier iterations are preserved as historical artifacts and are not the main result. See the [Experiment History](history/README.md) for V1–V3 reports, configurations, and curated results.

## Limitations

- Fifteen handcrafted scenarios are an exploratory systems evaluation, not a general benchmark.
- There was one deterministic model attempt per scenario.
- Diagnosis accuracy uses a frozen substring rubric.
- The AST allowlist measures contract adherence as well as behavioral quality.
- Held-out metrics cover final development-selected harnesses, not every rejected or unselected mutation.
- CPU latency does not predict GPU serving performance.
- The results do not reproduce GAIA2, SWE-Bench Pro, Terminal-Bench, or the paper's Claude experiments.
