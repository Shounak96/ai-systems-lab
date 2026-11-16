# Experiment 002 V4: AutoSaddler V2 local-model evaluation

## Design

This preregistered, exploratory evaluation used 15 independent handcrafted harness failures. Each scenario had one training case, two development cases, and two post-selection held-out cases. The held-out cases were absent from AutoSaddler `ScenarioComponents`, optimizer prompts/evidence, resolved development entities, acceptance, and ranking; they were evaluated only after `RunCompleted` against the engine-selected harness and the seed baseline.

The official AutoSaddler V2 engine was pinned at `e841543ce6cf6bdf690e65134b12a829ec9f66d1`. The frozen protocol digest was `sha256:93b7ed51eec9740014e6249cdf36170e9f7871a57c5dd2414546a0beb7f0a00b`. Both Q4_K_M models used temperature 0, seed 42, `num_ctx=4096`, `num_predict=1200`, thinking disabled, and zero model retries. One independent model attempt was run per scenario because another full trial would require 30 additional CPU calls.

## Aggregate comparison

| Metric, n=15 scenarios/model | Qwen3 1.7B | Qwen3 4B |
|---|---:|---:|
| Schema valid | 15/15, 100% [79.6%, 100%] | 15/15, 100% [79.6%, 100%] |
| Fixed-rubric diagnosis accurate | 10/15, 66.7% [41.7%, 84.8%] | 15/15, 100% [79.6%, 100%] |
| Mutation valid | 10/15, 66.7% [41.7%, 84.8%] | 12/15, 80.0% [54.8%, 93.0%] |
| Training accepted | 7/15, 46.7% [24.8%, 69.9%] | 12/15, 80.0% [54.8%, 93.0%] |
| Development improved | 4/15, 26.7% [10.9%, 52.0%] | 8/15, 53.3% [30.1%, 75.2%] |
| Development regression | 0/15, 0% [0%, 20.4%] | 2/15, 13.3% [3.7%, 37.9%] |
| Selected harness improved held-out score | 5/15, 33.3% [15.2%, 58.3%] | 8/15, 53.3% [30.1%, 75.2%] |
| Selected harness passed all held-out cases | 5/15, 33.3% [15.2%, 58.3%] | 8/15, 53.3% [30.1%, 75.2%] |
| Selected-harness held-out regression | 0/15, 0% [0%, 20.4%] | 0/15, 0% [0%, 20.4%] |

Brackets are two-sided Wilson 95% intervals with the scenario as the independent unit. Intervals are wide and overlap; the observed difference is descriptive evidence, not a statistical-significance claim.

## Scenario outcomes

Held-out deltas compare the final engine-selected harness with the seed.

| Scenario | 1.7B outcome; held-out Δ | 4B outcome; held-out Δ |
|---|---|---|
| Structured output | Evaluated training failure; 0 | Accepted, development regression, seed selected; 0 |
| Tool arguments | Verifier rejection; 0 | Accepted; +0.5 |
| Retries | Accepted; +0.5 | Accepted; +0.5 |
| State management | Verifier rejection; 0 | Accepted; +1.0 |
| Error handling | Accepted; +0.5 | Verifier rejection; 0 |
| Pagination | Accepted; +0.5 | Accepted; +0.5 |
| Canonical cache keys | Verifier rejection; 0 | Verifier rejection; 0 |
| Timeout budgeting | Accepted; 0 (seed already passed held-out) | Accepted; 0 (seed already passed held-out) |
| Event deduplication | Accepted; +0.5 | Accepted; +0.5 |
| Permission scopes | Verifier rejection; 0 | Verifier rejection; 0 |
| Fallback selection | Evaluated training failure; 0 | Accepted; 0 |
| Rate-limit delay | Accepted; 0 | Accepted with development regression; +0.5 |
| Completion tracking | Accepted; +0.5 | Accepted; +0.5 |
| Idempotency headers | Verifier rejection; 0 | Accepted; +0.5 |
| Status normalization | Evaluated training failure; 0 | Accepted; 0 |

The official development ranking selected the seed instead of the regressed 4B structured-output mutation. No selected harness regressed held-out score. Held-out evaluation covered selected harnesses only, not every rejected or unselected candidate.

## Inference cost

| Metric | Qwen3 1.7B | Qwen3 4B |
|---|---:|---:|
| Prompt tokens | 7,048 | 6,958 |
| Generated tokens | 2,820 | 3,410 |
| Total Ollama latency | 229.991 s | 570.997 s |
| Mean latency | 15.333 s | 38.066 s |
| p95 latency | 18.340 s | 42.317 s |
| Positive held-out improvements per inference minute | 1.304 | 0.841 |
| Total tokens per positive held-out improvement | 1,974 | 1,296 |

The 4B model produced more held-out improvements and used fewer total tokens per improvement, but its mean latency was 2.48× higher on this CPU. The 1.7B model produced 1.55× more held-out improvements per inference minute.

## Verification and failures

- The five-scenario gate passed before continuation: ten one-shot records, no provider errors, byte-identical normalized requests within every model pair, completed official-engine runs, and complete post-selection evaluations.
- Final audit reconciled 30 ledger rows with raw Ollama counters, official session/evaluation/acceptance events, selected-candidate events, and both held-out evaluations.
- Eight mutations were rejected before behavioral evaluation: five from 1.7B and three from 4B. Causes included forbidden loops/calls/imports, a no-op mutation, and operations outside the frozen allowlist.
- Three verifier-valid mutations failed training; all were from 1.7B.
- Two accepted 4B mutations regressed at least one development case. Development ranking prevented a held-out regression in the selected harnesses.
- The deterministic storage-only retry resolved transient Windows projection replacement locks. It never retried model calls or altered engine decisions. All 27 pre-inference tests passed, and the official checkout remained unmodified.

## Limitations and interpretation

These are 15 synthetic scenarios, one seed, and one model attempt per scenario—not a benchmark or a reproduction of AutoSaddler's Claude results. The diagnosis rubric uses frozen substrings and may undercount semantically adequate explanations. The AST allowlist measures contract compliance as well as code quality. Held-out cases measure final development-selected harnesses, so they assess the end-to-end workflow rather than every generated mutation. CPU latency may not generalize to GPU deployments.

The evidence supports a cautious public technical post about this bounded experiment: 4B achieved more absolute generalizing improvements, while 1.7B was faster and more time-efficient. A post must publish the preregistration, raw artifacts, wide confidence intervals, single-trial limitation, and the absence of statistical-significance or paper-equivalence claims.

Charts: `artifacts/v4/charts/quality_rates.svg` and `artifacts/v4/charts/latency.svg`.
