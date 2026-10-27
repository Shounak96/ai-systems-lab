# 001 - Semantic Cache Guardrails

## Problem

Embedding-based caches can reuse an answer for a paraphrased query, but high semantic similarity does not guarantee that the cached answer remains correct. Small changes to a number, date, negation, version, tenant, or operational condition may require a different response.

This experiment asks whether lightweight deterministic query-side checks can reduce incorrect semantic-cache reuse while preserving useful hits and reasonable lookup latency.

## Research inspiration

The work is inspired by [LaCache: Robust Semantic Caching for LLM Serving](https://arxiv.org/html/2608.01718v2) (Liang et al., 2026).

This is an independent investigation, not a LaCache implementation. LaCache verifies a freshly generated response prefix and integrates draft-model/speculative decoding techniques. This repository implements query-side attribute checks only and does not inherit LaCache's formal or adversarial guarantees.

## Implemented strategies

| Strategy | Decision path |
|---|---|
| Exact match | Conservative whitespace/case normalization, then partitioned key lookup |
| Semantic cache | MiniLM query embedding, scope-eligible nearest match, then cosine threshold |
| Guarded semantic cache | Semantic decision plus configurable checks for numbers, dates, negation/polarity, versions, and selected technical entities |

Tenant and user scope isolation is an access-control boundary applied before semantic matching; it is not an optional guard.

The implementation is intentionally in memory and uses a transparent linear scan. It supports configurable thresholds, LRU capacity, structured lookup outcomes, component latency measurements, and auditable guard-rejection reasons.

## Architecture

```text
incoming query
    |
    +-- scope / tenant partition
    |
    +-- normalized exact key --------------------------> exact hit or miss
    |
    +-- MiniLM embedding -> nearest cached query
                              |
                              +-- cosine threshold ----> semantic hit or miss
                              |
                              +-- attribute guards ----> guarded hit or miss
```

Key modules:

- [`cache.py`](src/semantic_cache/cache.py): exact, semantic, and guarded caches.
- [`guards.py`](src/semantic_cache/guards.py): deterministic attribute extraction and comparison.
- [`evaluate_v2.py`](src/semantic_cache/evaluate_v2.py): controlled benchmark, metrics, bootstrap intervals, and charts.
- [`verify_v2.py`](src/semantic_cache/verify_v2.py): independent reconciliation of summaries with raw CSV artifacts.

## Dataset methodology

[`test_cases_v2.csv`](data/test_cases_v2.csv) contains 240 labeled cases in 38 query families:

- 228 manually authored comparisons for APIs, deployment, permissions, tenants, versions, service state, storage, and operations;
- 12 disclosed exact repeats materialized from curated seeds;
- 126 reuse-eligible positives and 114 ineligible negatives;
- 58 development cases and 182 held-out cases.

Each row records its family, split, comparison seed, reuse label, category, provenance, explanation, and scope metadata. [`build_dataset_v2.py`](data/build_dataset_v2.py) deterministically regenerates and validates the dataset. Related variants remain in the same split to prevent family leakage.

This is a manually constructed research benchmark, not production traffic or an external real-world dataset.

## Evaluation methodology

- Fixed random seed: `20261023`.
- Thresholds: 0.70, 0.85, 0.90, and 0.95.
- Identical query order and declared seed entry across strategies.
- Fresh one-entry cache per case; evaluation queries do not populate it.
- Warm CPU execution with three measured lookups per case/configuration.
- Semantic timings include incoming-query embedding, search, and applicable guard work.
- Model load, first warmup, and seed insertion are reported separately.
- Confidence intervals resample independent query families, not related variants.
- The selected configuration is chosen on development data and reported on the held-out split.

No LLM inference is included, so lookup timings are not end-to-end response latency.

## Held-out results

At the development-selected threshold of 0.95:

| Metric | Semantic baseline | Guarded cache |
|---|---:|---:|
| Correct hits | 22 | 20 |
| Incorrect hits | 32 | 6 |
| Correct-hit precision | 40.74% | 76.92% |
| Unsafe reuse rate | 17.58% | 3.30% |
| Mean lookup latency | 9.849 ms | 10.253 ms |

The guards reduced incorrect hits by **81.25%**, while losing **2 correct hits**. Direct guard work averaged 0.117 ms; the separately executed guarded configuration had approximately 0.404 ms higher mean total lookup latency.

Six unsafe accepts remained, involving environment, namespace, feature state, health state, and before/after relationships. The guarded cache therefore improved this benchmark but did not make semantic caching safe in general.

See the [full report](REPORT_V2.md), [raw summary](results_v2/summary_metrics.json), [predictions](results_v2/predictions.csv), and [charts](results_v2/).

## Installation

Python 3.11 or 3.12 is recommended. From this directory in PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The first benchmark run downloads `sentence-transformers/all-MiniLM-L6-v2`. Model weights are stored locally and excluded from Git.

## Reproduce Version 2

```powershell
$env:PYTHONPATH = "src;."
$env:HF_HOME = "$PWD\.cache\huggingface"
$env:MPLCONFIGDIR = "$PWD\.cache\matplotlib"
$env:TEMP = "$PWD\.tmp-v2"
$env:TMP = $env:TEMP

.\.venv\Scripts\python.exe data\build_dataset_v2.py
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider tests
.\.venv\Scripts\python.exe -m semantic_cache.evaluate_v2
.\.venv\Scripts\python.exe -m semantic_cache.verify_v2
```

After the model has been cached, fully offline execution can also set:

```powershell
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
```

## Results and historical artifacts

- [`REPORT_V2.md`](REPORT_V2.md): methodology, metrics, uncertainty, error analysis, and limitations.
- [`results_v2/`](results_v2/): raw predictions, latency samples, summaries, environment metadata, and five charts.
- [`archive/v1/`](archive/v1/): preserved Version 1 report, dataset, results, and chart.

## Limitations

- The balanced benchmark was constructed for this experiment and does not estimate production error prevalence.
- Only 38 query families, one embedding model, one machine, and a one-entry in-memory cache were evaluated.
- The finite English guard vocabulary can reject valid paraphrases and miss domain-specific state changes.
- Large-index retrieval, concurrency, persistence, expiration, distributed enforcement, and production traffic were not tested.
- No response-prefix verification, adversarial prompt optimization, LLM generation, token accounting, or cost measurement was implemented.

The strongest supported conclusion is narrow: deterministic query-side checks substantially reduced incorrect reuse on this disclosed benchmark, but thresholding and lightweight guards still left unsafe cache hits.
