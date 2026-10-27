# Experiment report

Run date: 2026-10-23  
Embedding model: `sentence-transformers/all-MiniLM-L6-v2`  
Model revision downloaded: `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`  
Execution: CPU, Python 3.12.10, no LLM inference

## What was implemented

- A reusable sentence-transformer embedding adapter.
- Normalized exact-match and cosine-similarity semantic response caches.
- Configurable inclusive similarity thresholds and best-match lookup.
- Explicit global/user partitions and rejection of user-specific data in global scope by default.
- Optional LRU capacity, structured debug logging, and per-query timing.
- Fifteen manually constructed, explicitly labeled cases: one normalized exact repeat, four benign paraphrases, and ten contrasts involving numbers, negation, entities, or user context.
- Eight automated tests for normalization, hits/misses, inclusive threshold behavior, false reuse, scope isolation, privacy controls, LRU eviction, and label-based evaluation.

## Commands used

From `experiments/001-semantic-cache` in PowerShell:

```powershell
py -3.12 -m venv .venv
$env:TEMP = "$PWD\.tmp"
$env:TMP = $env:TEMP
.\.venv\Scripts\python.exe -m ensurepip --upgrade --default-pip
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:PYTHONPATH = "src"
$env:HF_HOME = "$PWD\.cache\huggingface"
$env:MPLCONFIGDIR = "$PWD\.cache\matplotlib"
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m semantic_cache.evaluate
```

## Actual results

The test suite passed: **8 passed**.

| Strategy | Threshold | Hits / 15 | Hit rate | Correct / incorrect reuse | Reuse precision | False-hit rate on 10 contrasts | Mean lookup pipeline |
|---|---:|---:|---:|---:|---:|---:|---:|
| Exact | — | 1 | 6.7% | 1 / 0 | 100% | 0% | 0.011 ms |
| Semantic | 0.70 | 12 | 80.0% | 5 / 7 | 41.7% | 70% | 11.049 ms |
| Semantic | 0.75 | 10 | 66.7% | 5 / 5 | 50.0% | 50% | 8.412 ms |
| Semantic | 0.80 | 10 | 66.7% | 5 / 5 | 50.0% | 50% | 9.334 ms |
| Semantic | 0.85 | 10 | 66.7% | 5 / 5 | 50.0% | 50% | 8.997 ms |
| Semantic | 0.90 | 8 | 53.3% | 4 / 4 | 50.0% | 40% | 8.117 ms |
| Semantic | 0.95 | 4 | 26.7% | 2 / 2 | 50.0% | 20% | 8.544 ms |

These are lookup pipeline times on this machine. Semantic timings include candidate-query embedding and a one-entry in-memory scan, but exclude model loading, seed insertion, and any answer generation. Exact timings are not directly comparable to full LLM latency.

The highest-similarity incorrect reuses were:

- Negation: “Should I enable automatic backups?” vs. “Should I not enable automatic backups?” — cosine similarity **0.978085**.
- Changed year: February 2024 vs. February 2023 — **0.961155**.
- Changed quantity: 10 km vs. 100 km — **0.927828**.
- Negation: foods that contain vs. do not contain peanuts — **0.926811**.

User-context cases did not cross scopes, even when the text was identical. Full observations, including the exact queries and hit decisions, are in `results/raw_results.csv`.

## Failures and unexpected behavior

- Initial `venv` bootstrapping and pytest cache cleanup hit Windows permission errors in the managed default temporary directory. Redirecting `TEMP`, model caches, and matplotlib configuration into ignored experiment-local directories resolved the benchmark path. Tests passed despite one cache-cleanup warning on the first run.
- Raising the threshold did not steadily improve reuse precision in this tiny dataset. It removed both valid paraphrases and invalid contrasts; precision stayed at 50% from 0.75 through 0.95.
- A 0.95 threshold was still not sufficient to prevent high-similarity errors caused by negation and a changed year.

## Engineering trade-offs

- Exact matching was conservative and extremely cheap, but missed all four non-identical paraphrases.
- Semantic matching recovered all four paraphrases through threshold 0.85, but also reused answers for five of eight same-scope contrasts at those thresholds. Scope isolation independently prevented the two cross-user cases.
- Threshold tuning alone cannot encode whether a changed token is answer-critical. Higher thresholds reduce utility and still leave some severe errors.
- The current linear scan is transparent and sufficient for 15 pairwise trials. A production cache would need an approximate index, concurrency control, persistence, expiry/invalidation, observability, and a much broader validation set.

## Relationship to LaCache

Implemented from the motivating problem: query embedding, cosine-threshold cache decisions, benign reuse, incorrect reuse, and the efficiency/correctness tension.

Not implemented: cache-collision attacks, response-prefix lookahead, LSH, a draft or backbone LLM, speculative decoding, multiple lookahead checks, adaptive attacks, security proofs, the paper's datasets, or its production cost/latency analysis. Therefore these results neither reproduce nor validate LaCache's reported security or utility metrics.

## Findings suitable for public discussion

The strongest defensible observation is narrow: **on this disclosed 15-case manual probe set, a general-purpose sentence embedding assigned very high cosine similarity to several query pairs whose correct answers differ, and threshold increases alone did not eliminate incorrect reuse without also losing benign reuse**. The explicit 0.978 negation and 0.961 changed-year examples are concrete and reproducible.

The exact rates are not strong enough to generalize beyond this dataset. They should not be described as production failure rates, an external benchmark, a LaCache reproduction, or evidence about adversarial robustness. A larger preregistered dataset and repeated latency trials would be needed before making broader claims.
