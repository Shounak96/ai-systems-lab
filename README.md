# AI Systems Lab

Research-driven AI and software engineering experiments exploring reliability, correctness, performance, and real-world trade-offs through hands-on implementations and reproducible benchmarks.

This repository turns research questions into small, inspectable systems. Each experiment defines a narrow hypothesis, implements competing approaches, evaluates them on disclosed data, preserves raw measurements, and documents both useful findings and failure cases.

## Methodology

1. Start with a concrete systems question motivated by research or engineering practice.
2. Define baselines, controls, correctness labels, and measurable outcomes.
3. Build a minimal implementation that exposes the relevant trade-offs.
4. Run reproducible tests and benchmarks using recorded environments and fixed seeds.
5. Publish raw artifacts, analysis, limitations, and exact reproduction commands.

## Current experiments

### 001 - Semantic Cache Guardrails

Can lightweight query-side validation reduce incorrect semantic-cache reuse without eliminating useful cache hits?

The experiment compares normalized exact matching, embedding-based semantic caching, and a guarded semantic cache that checks correctness-sensitive attributes such as numbers, dates, negation, software versions, and selected technical entities.

At the development-selected threshold on **182 held-out cases**:

- incorrect hits fell from **32 to 6**, an **81.25% reduction**;
- correct-hit precision increased from **40.74% to 76.92%**;
- safeguards lost **2 correct hits** relative to the semantic baseline;
- mean total cache-lookup overhead was approximately **0.404 ms**.

These measurements come from a manually constructed benchmark and local, one-entry cache workload. They do not represent production traffic, end-to-end LLM latency, or a security guarantee.

[Experiment overview](experiments/001-semantic-cache/README.md) · [Full V2 report](experiments/001-semantic-cache/REPORT_V2.md) · [Strategy comparison](experiments/001-semantic-cache/results_v2/03_strategy_comparison.png) · [Latency chart](experiments/001-semantic-cache/results_v2/04_lookup_latency.png) · [Raw results](experiments/001-semantic-cache/results_v2/summary_metrics.json)

### 002 - AutoSaddler Local-Model Harness Optimization

Can inexpensive local Qwen models perform AutoSaddler V2's diagnosis and structured harness-modification role, and do accepted changes generalize to held-out failures?

Across 15 preregistered scenarios, Qwen3 1.7B improved the final held-out score in **5/15** scenarios, while Qwen3 4B improved it in **8/15**. The 4B model produced more absolute improvements, but its mean CPU inference latency was **2.48× higher**. Wilson confidence intervals were wide and overlapping, so the comparison is descriptive rather than a statistical-superiority claim.

The experiment uses the pinned official AutoSaddler V2 engine, local Ollama inference, structured component updates, hidden post-selection tests, raw-event auditing, and zero paid API calls.

[Experiment overview](experiments/002-agent-patch-quality/README.md) · [Technical report](experiments/002-agent-patch-quality/V4_TECHNICAL_REPORT.md) · [Audited results](experiments/002-agent-patch-quality/artifacts/v4/results.json) · [Quality chart](experiments/002-agent-patch-quality/artifacts/v4/charts/quality_rates.svg) · [Experiment history](experiments/002-agent-patch-quality/history/README.md)

## Technologies and engineering areas

- Python 3.12, pytest, NumPy, and Matplotlib
- Sentence Transformers and cosine-similarity retrieval
- Cache correctness, tenant isolation, and conservative validation
- Benchmark design, latency decomposition, and bootstrap uncertainty
- Reproducible datasets, raw measurement artifacts, and regression testing

## Reproducibility

Experiments keep source code, dataset builders, labeled evaluation data, environment metadata, raw predictions, timing samples, charts, and reports together. Generated model weights, virtual environments, secrets, and disposable caches are excluded from Git.

Start with the reproduction section in each experiment README. Results should be interpreted within the stated workload and limitations; synthetic or manually authored cases are never presented as production benchmarks.

## Future experiments

Future work will continue examining practical AI-system behavior, including retrieval quality, evaluation reliability, inference serving, observability, agent workflows, and software architecture trade-offs. New work will be added as numbered, self-contained experiments under `experiments/` or as larger implementations under `projects/`.
