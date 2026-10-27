# Version 2 report: Guarded semantic caching

- Run date: 2026-10-23
- Status: completed and independently reconciled from raw artifacts
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- Execution: CPU, Python 3.12.10, no LLM inference

## 1. Research question

Can lightweight deterministic checks on correctness-sensitive query attributes reduce unsafe semantic-cache reuse while preserving useful hits and reasonable local lookup latency?

## 2. Hypothesis

Numbers, dates, explicit negation, software versions, and reliably recognized technical entities explain a meaningful subset of high-similarity but non-equivalent queries. Rejecting a semantic hit when these attributes conflict should reduce incorrect reuse. The expected cost is extra lookup latency and false rejection of some valid paraphrases.

## 3. Research-paper connection

This is an independent investigation inspired by *LaCache: Robust Semantic Caching for LLM Serving* (Liang et al., 2026): https://arxiv.org/html/2608.01718v2.

LaCache addresses cache-collision attacks with a second, response-side check over freshly generated response-prefix tokens, using a draft LLM and speculative decoding. This experiment does not implement that architecture. It studies deterministic query-side checks only and makes no adversarial, formal-security, or production-readiness claim.

## 4. Version 1 audit and metric correction

Version 1 is preserved unchanged under `archive/v1/`.

- Exact matching used case-folding plus whitespace normalization. Semantic matching embedded the incoming query and linearly scanned the scope-eligible entries for the highest cosine similarity.
- Each case created a fresh cache, inserted one declared seed, and issued one query. Queries never populated the cache. Results therefore did not depend on case order or cross-case warming; model process warming could affect timing.
- Labels were manual `acceptable_reuse` booleans with explicit expected answers. Correctness did not use similarity as a proxy.
- Semantic latency included incoming-query embedding and lookup, while seed embedding and model loading were excluded.
- User-scope partitions were mandatory and prevented comparisons across user IDs.

The apparent V1 discrepancy was caused by two different denominators:

- Incorrect-hit rate among hits: `incorrect hits / all hits`. At threshold 0.70 this was `7 / 12 = 58.3%`.
- Negative-case false-hit rate, called "false-hit rate on contrasts" in V1: `incorrect hits / all ground-truth negative cases`. It was `7 / 10 = 70%`.

Both calculations were valid, but V1 reported only the second and the prose made it easy to interpret as the first. V2 reports both explicitly, plus unsafe reuse over all queries.

## 5. Dataset construction and labeling

The V2 dataset contains **240 cases in 38 independent query families**:

- 228 manually authored comparisons covering software APIs, deployment, access control, tenants, versions, availability, observability, storage, and operational policy.
- 12 programmatic exact repeats copied from curated family seeds. These are explicitly marked `programmatic_exact_repeat_from_curated_seed`; they are 5% of the dataset.
- 126 reuse-eligible positives and 114 ineligible negatives.
- 58 development cases and 182 held-out test cases.

Every row contains a unique query ID, family ID, split, query, intended seed, synthetic response identifier, eligibility label, category, provenance, explanation, and scope metadata. No external dataset was used, so there is no external version or license to report. The synthetic response identifiers only test cache routing; they are not LLM answers.

The text and labels are deliberately constructed research fixtures, not production traffic or an external benchmark. The generator performs no random language generation. A fixed seed (`20261023`) assigns whole families to development or test, preventing related variants from crossing the split. `data/build_dataset_v2.py` validates uniqueness, labels, provenance, required scope IDs, split coverage, and family isolation.

Positive cases include paraphrases, word-order changes, technical equivalents, synonyms, number/version formatting, and exact repeats. Negative cases change dates, quantities, negation, products, tenants, permissions, versions, environments, identities, service states, and operation order.

## 6. Implemented approaches

### A. Exact-match baseline

Conservative whitespace and case normalization followed by partitioned key lookup.

### B. Standard semantic baseline

L2-normalized MiniLM embeddings, cosine similarity, best scope-eligible match, and inclusive thresholds of 0.70, 0.85, 0.90, and 0.95.

### C. Guarded semantic cache

The same semantic retrieval runs first. A hit above threshold is accepted only if enabled deterministic guards do not find conflicts in:

- numeric tokens, with commas normalized and date/version spans handled separately;
- calendar dates and years;
- explicit negation and selected operational polarity terms;
- software versions, including common `v2`, `version 2`, and product-major forms;
- a finite, documented catalog of technical product entities and aliases.

Each guard is independently configurable. Rejection means "do not reuse because query-side evidence is ambiguous or conflicting," not "the queries are proven different." Tenant/user partitioning occurs before retrieval and is not an optional semantic guard.

## 7. Benchmark configuration and controls

- Model: `sentence-transformers/all-MiniLM-L6-v2`, CPU.
- Fixed dataset/sampling seed: 20261023.
- Identical case order, seed entry, scope, cache capacity, and query set for all strategies.
- Pairwise fresh cache with capacity 1; no eviction occurs during a case.
- Queries do not update or populate the cache.
- Seed insertion and its embedding are outside lookup timing.
- Incoming-query embedding, candidate search, and guard work are inside lookup timing and recorded separately.
- Three measured warm lookups per case/configuration; model load and first warmup are recorded separately.
- Model load: 7,277.24 ms. First warmup embedding: 17.09 ms.
- Throughput is `1000 / mean lookup milliseconds`, a derived sequential local lookup service rate. It is not concurrent server throughput.
- No LLM inference, network cache, answer-generation time, token savings, or monetary savings are measured.
- Guard threshold selection used development data only: maximize safe-hit coverage among configurations with development unsafe-reuse rate at most 2%; break ties by precision and then lower threshold. Threshold 0.95 qualified with 1/58 unsafe reuse and was selected.

During preflight, development/unit checks exposed version canonicalization, sentence-final number parsing, entity alias, and polarity-token defects. These deterministic implementation bugs were fixed and regression-tested before the final saved run. No held-out labels were changed.

## 8. Metric definitions

For `N` evaluated queries, `H` hits, `C` correct hits, `I` incorrect hits, `E` eligible queries, and `G` ineligible queries:

| Metric | Numerator / denominator |
|---|---|
| Hit rate | `H / N` |
| Correct-hit precision | `C / H` |
| Incorrect-hit rate among hits | `I / H` |
| Unsafe reuse rate | `I / N` |
| Safe-hit coverage | `C / N` |
| Reuse-eligibility recall | `C / E` |
| False-rejection rate | `(E - C) / E` |
| Negative-case false-hit rate | `I / G` |

An eligible query is a false rejection whenever it is not correctly reused, whether because similarity is below threshold or a guard rejects it.

## 9. Actual held-out results

The held-out test set has 182 queries: 95 eligible and 87 ineligible.

| Strategy | Threshold | Hits | Correct | Incorrect | Precision | Unsafe reuse | Safe coverage | Eligibility recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Exact | - | 8 | 8 | 0 | 100.0% | 0.0% | 4.40% | 8.42% |
| Semantic | 0.70 | 178 | 95 | 83 | 53.37% | 45.60% | 52.20% | 100.0% |
| Guarded | 0.70 | 123 | 87 | 36 | 70.73% | 19.78% | 47.80% | 91.58% |
| Semantic | 0.85 | 137 | 70 | 67 | 51.09% | 36.81% | 38.46% | 73.68% |
| Guarded | 0.85 | 91 | 65 | 26 | 71.43% | 14.29% | 35.71% | 68.42% |
| Semantic | 0.90 | 108 | 53 | 55 | 49.07% | 30.22% | 29.12% | 55.79% |
| Guarded | 0.90 | 66 | 50 | 16 | 75.76% | 8.79% | 27.47% | 52.63% |
| Semantic | 0.95 | 54 | 22 | 32 | 40.74% | 17.58% | 12.09% | 23.16% |
| Guarded, selected | 0.95 | 26 | 20 | 6 | 76.92% | 3.30% | 10.99% | 21.05% |

At the selected 0.95 threshold, guards reduced incorrect hits from 32 to 6: **26 fewer, or an 81.25% reduction** relative to standard semantic caching. Correct hits decreased from 22 to 20: **2 valid hits lost**. Precision increased from 40.74% to 76.92%. The method did not eliminate unsafe reuse.

The lower 0.70 threshold provides the clearest utility-oriented trade-off: guards reduced incorrect hits from 83 to 36 (56.63%) while reducing correct hits from 95 to 87 (8 lost).

## 10. Statistical uncertainty

Percentile 95% bootstrap intervals use 2,000 resamples of independent families, not individual query variants.

For the selected guarded 0.95 configuration:

- Correct-hit precision: 76.92%, 95% family-bootstrap CI **60.87%-90.48%**.
- Unsafe reuse rate: 3.30%, CI **1.10%-5.62%**.
- Safe-hit coverage: 10.99%, CI **6.11%-16.76%**.
- Reuse-eligibility recall: 21.05%, CI **11.83%-31.63%**.

The wide intervals reflect the modest number of independent families and the small number of hits at threshold 0.95.

## 11. Performance

| Strategy | Threshold | Mean | p50 | p95 | Mean embedding | Mean guard | Derived sequential QPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| Exact | - | 0.004 ms | 0.004 ms | 0.005 ms | 0 | 0 | 237,257 |
| Semantic | 0.95 | 9.849 ms | 10.765 ms | 11.927 ms | 9.817 ms | 0 | 101.53 |
| Guarded | 0.95 | 10.253 ms | 11.065 ms | 12.225 ms | 10.104 ms | 0.117 ms | 97.53 |

At the selected threshold, directly instrumented guard work averaged **0.117 ms** (p95 **0.466 ms**). Total mean lookup was 0.404 ms higher than the separately executed semantic run, and derived sequential throughput was 3.94% lower. Embedding time dominates and normal CPU scheduling variation affects cross-configuration latency comparisons; direct guard timing is the cleaner overhead measure.

These are warm, one-entry, in-memory lookup measurements. They do not represent end-to-end LLM response time or behavior at index scale.

## 12. Error analysis and unexpected findings

At threshold 0.95, guards rejected otherwise high-similarity negative cases through 13 numeric mismatches, 6 version mismatches, 3 polarity mismatches, 3 negation mismatches, and 1 date mismatch. Reasons can overlap.

Two eligible cases were rejected by guards rather than threshold:

- `terraform_region-01`: the entity catalog treated `AWS` and the absence of the explicit token in the paraphrase differently.
- `webhook-03`: `four` versus `4` triggered the conservative numeric check.

This confirms the expected false-rejection trade-off and shows why a query-side extractor needs canonicalization or domain schemas.

Six unsafe accepts remained at the selected configuration:

- production vs development deployment environment (`replicas-06`, similarity 0.979702);
- Kubernetes namespace `prod` vs `staging` (`kubernetes_namespace-05`, 0.974096);
- new vs legacy feature (`feature_flag-06`, 0.957350);
- healthy vs unhealthy service state (`health_check-05`, 0.955553);
- migration before vs after deployment (`migration-06`, 0.964362);
- cache invalidation before vs after update (`cache_invalidation-04`, 0.986178).

The remaining failures concern domain state, identifiers, and temporal/operational relations outside the implemented guard vocabulary. Identical entities and numbers did not imply answer equivalence.

Unexpectedly, increasing the semantic threshold alone reduced recall while standard-cache precision fell from 53.37% at 0.70 to 40.74% at 0.95. In this constructed set, some of the hardest negative pairs were more lexically similar than genuine paraphrases.

## 13. Engineering trade-offs

- Exact matching was safe and extremely cheap but covered only 8 of 95 reuse opportunities.
- Standard semantic caching maximized reuse but admitted many answer-changing pairs.
- Guards improved precision at every tested threshold and had small direct CPU cost, but they introduced false rejection and still missed relational changes.
- Conservative mismatches are appropriate when a wrong response is costly, but they may erase much of semantic caching's utility at a high threshold.
- Domain-specific structured keys could handle environments, namespaces, roles, endpoints, and operation ordering more reliably than a generic regex layer.
- Query-side checks remain manipulable and do not provide LaCache's response-side security argument.

## 14. Threats to validity and limitations

- The benchmark is manually constructed and balanced by design; rates do not estimate production prevalence.
- Families are diverse but only 38 are statistically independent.
- Labels were authored for this experiment and were not independently adjudicated.
- The test split was not used for threshold selection, but final error analysis necessarily inspects it.
- The one-entry linear cache does not test nearest-neighbor competition, large indexes, concurrency, eviction, persistence, or distributed scope enforcement.
- Only one embedding model, one machine, one run date, and three latency repeats were used.
- The finite entity lexicon is incomplete and English-specific.
- No LLM inference, adversarial optimization, response-prefix verification, speculative decoding, cost modeling, or token measurement is included.
- Bootstrap intervals quantify family-sampling uncertainty within this constructed benchmark, not uncertainty about real traffic.

## 15. Reproduction

From `experiments/001-semantic-cache` in PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
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

The first model use requires network access unless the model is already cached. For a cached model, set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`.

Primary artifacts:

- `results_v2/predictions.csv`: every hit/miss, similarity, and guard decision.
- `results_v2/latency_measurements.csv`: 6,480 component timing rows.
- `results_v2/summary_metrics.json`: definitions, selected threshold, counts, rates, intervals, and latency.
- `results_v2/category_metrics.json`: per-category breakdown.
- `results_v2/environment.json`: hardware/software and cold/warm metadata.
- `results_v2/01_...png` through `05_...png`: denominator-labeled charts.

`semantic_cache.verify_v2` independently recomputed counts and timing aggregates from the raw CSV files and verified all 18 summaries and five charts. The final run produced 2,160 prediction rows, 6,480 timing rows, and **28 passing tests**.

## 16. Difference from LaCache

This experiment implements query embedding plus deterministic query-attribute comparisons. LaCache performs a response-side lookahead check over newly generated prefix tokens and analyzes robustness against cache-collision attacks. This V2 work does not reproduce LaCache, validate its reported results, or inherit its formal guarantees. It is an independent, small-scale engineering study motivated by the same unsafe-reuse problem.
