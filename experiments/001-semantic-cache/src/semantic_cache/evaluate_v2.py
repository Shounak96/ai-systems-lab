"""Rigorous V2 evaluation for exact, semantic, and guarded semantic caches."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import importlib.metadata
import json
import logging
import math
from pathlib import Path
import platform
import random
from statistics import mean, median
from time import perf_counter_ns

import matplotlib.pyplot as plt

from .cache import ExactMatchCache, GuardedSemanticCache, SemanticCache
from .embedding import SentenceTransformerEmbedder

RANDOM_SEED = 20261023
BOOTSTRAP_SAMPLES = 2000


@dataclass
class Prediction:
    split: str
    strategy: str
    threshold: str
    query_id: str
    family_id: str
    category: str
    reuse_eligible: bool
    hit: bool
    correct_hit: bool
    incorrect_hit: bool
    false_rejection: bool
    similarity: str
    candidate_query: str
    guard_passed: str
    guard_reasons: str
    query: str
    seed_query: str


@dataclass
class Measurement:
    split: str
    strategy: str
    threshold: str
    query_id: str
    family_id: str
    repeat: int
    total_ms: float
    embedding_ms: float
    search_ms: float
    guard_ms: float


def read_cases(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "query_id", "family_id", "split", "query", "seed_query", "seed_response",
        "expected_response", "reuse_eligible", "category", "provenance",
        "label_explanation", "seed_scope", "seed_user_id", "query_scope",
        "query_user_id", "contains_user_data",
    }
    if not rows or not required.issubset(rows[0]):
        raise ValueError("V2 dataset is empty or missing required columns")
    if len({row["query_id"] for row in rows}) != len(rows):
        raise ValueError("query_id values must be unique")
    family_splits: dict[str, set[str]] = {}
    for row in rows:
        family_splits.setdefault(row["family_id"], set()).add(row["split"])
    leaked = [family for family, splits in family_splits.items() if len(splits) != 1]
    if leaked:
        raise ValueError(f"families occur in multiple splits: {leaked}")
    return rows


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    rank = (len(ordered) - 1) * percentile
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower)


def run_configuration(
    cases: list[dict[str, str]],
    strategy: str,
    threshold: float | None,
    embedder,
    repeats: int,
) -> tuple[list[Prediction], list[Measurement]]:
    predictions: list[Prediction] = []
    measurements: list[Measurement] = []
    threshold_text = "" if threshold is None else f"{threshold:.2f}"
    for case in cases:
        if strategy == "exact":
            cache = ExactMatchCache(max_size=1)
        elif strategy == "semantic":
            cache = SemanticCache(embedder, threshold=threshold, max_size=1)
        elif strategy == "guarded":
            cache = GuardedSemanticCache(embedder, threshold=threshold, max_size=1)
        else:
            raise ValueError(f"unknown strategy: {strategy}")

        seed_scope, query_scope = case["seed_scope"], case["query_scope"]
        seed_user, query_user = case["seed_user_id"] or None, case["query_user_id"] or None
        cache.put(
            case["seed_query"], case["seed_response"], scope=seed_scope,
            user_id=seed_user, contains_user_data=case["contains_user_data"] == "true",
        )
        outcomes = []
        for repeat in range(1, repeats + 1):
            if strategy == "exact":
                started = perf_counter_ns()
                hit = cache.get(case["query"], scope=query_scope, user_id=query_user)
                total_ns = perf_counter_ns() - started
                outcome = None
                embedding_ns = guard_ns = 0
                search_ns = total_ns
            else:
                outcome = cache.lookup(case["query"], scope=query_scope, user_id=query_user)
                hit = outcome.hit
                embedding_ns, search_ns, guard_ns = outcome.embedding_ns, outcome.search_ns, outcome.guard_ns
                total_ns = outcome.total_ns
            outcomes.append((hit is not None, "" if outcome is None or outcome.similarity is None else f"{outcome.similarity:.6f}"))
            measurements.append(Measurement(
                case["split"], strategy, threshold_text, case["query_id"], case["family_id"],
                repeat, total_ns / 1_000_000, embedding_ns / 1_000_000,
                search_ns / 1_000_000, guard_ns / 1_000_000,
            ))
        if len(set(outcomes)) != 1:
            raise AssertionError(f"non-deterministic decision for {case['query_id']}: {outcomes}")

        eligible = case["reuse_eligible"] == "true"
        is_hit = hit is not None
        correct = is_hit and eligible and hit.response == case["expected_response"]
        incorrect = is_hit and not eligible
        decision = None if strategy == "exact" else outcome.guard_decision
        predictions.append(Prediction(
            case["split"], strategy, threshold_text, case["query_id"], case["family_id"],
            case["category"], eligible, is_hit, correct, incorrect, eligible and not correct,
            "" if strategy == "exact" and not is_hit else ("1.000000" if strategy == "exact" else outcomes[0][1]),
            "" if strategy == "exact" or outcome.candidate_query is None else outcome.candidate_query,
            "" if decision is None else str(decision.accepted),
            "" if decision is None else "|".join(decision.reasons), case["query"], case["seed_query"],
        ))
    return predictions, measurements


def _counts(rows: list[Prediction]) -> dict[str, int]:
    return {
        "queries": len(rows),
        "eligible_queries": sum(row.reuse_eligible for row in rows),
        "ineligible_queries": sum(not row.reuse_eligible for row in rows),
        "hits": sum(row.hit for row in rows),
        "correct_hits": sum(row.correct_hit for row in rows),
        "incorrect_hits": sum(row.incorrect_hit for row in rows),
        "false_rejections": sum(row.false_rejection for row in rows),
    }


def _rates(counts: dict[str, int]) -> dict[str, float | None]:
    n, hits = counts["queries"], counts["hits"]
    eligible, ineligible = counts["eligible_queries"], counts["ineligible_queries"]
    correct, incorrect = counts["correct_hits"], counts["incorrect_hits"]
    return {
        "hit_rate": hits / n,
        "correct_hit_precision": correct / hits if hits else None,
        "incorrect_hit_rate_among_hits": incorrect / hits if hits else None,
        "unsafe_reuse_rate": incorrect / n,
        "safe_hit_coverage": correct / n,
        "reuse_eligibility_recall": correct / eligible if eligible else None,
        "false_rejection_rate": counts["false_rejections"] / eligible if eligible else None,
        "negative_case_false_hit_rate": incorrect / ineligible if ineligible else None,
    }


def bootstrap_intervals(rows: list[Prediction]) -> dict[str, list[float] | None]:
    by_family: dict[str, list[Prediction]] = {}
    for row in rows:
        by_family.setdefault(row.family_id, []).append(row)
    family_ids = sorted(by_family)
    rng = random.Random(RANDOM_SEED)
    sampled: dict[str, list[float]] = {name: [] for name in (
        "hit_rate", "correct_hit_precision", "incorrect_hit_rate_among_hits",
        "unsafe_reuse_rate", "safe_hit_coverage", "reuse_eligibility_recall",
    )}
    for _ in range(BOOTSTRAP_SAMPLES):
        replicate: list[Prediction] = []
        for family_id in rng.choices(family_ids, k=len(family_ids)):
            replicate.extend(by_family[family_id])
        rates = _rates(_counts(replicate))
        for name in sampled:
            if rates[name] is not None:
                sampled[name].append(float(rates[name]))
    return {
        name: [_percentile(values, 0.025), _percentile(values, 0.975)] if values else None
        for name, values in sampled.items()
    }


def summarize(
    predictions: list[Prediction], measurements: list[Measurement]
) -> list[dict[str, object]]:
    keys = sorted({(row.split, row.strategy, row.threshold) for row in predictions})
    summaries = []
    for split, strategy, threshold in keys:
        rows = [row for row in predictions if (row.split, row.strategy, row.threshold) == (split, strategy, threshold)]
        times = [row for row in measurements if (row.split, row.strategy, row.threshold) == (split, strategy, threshold)]
        counts = _counts(rows)
        rates = _rates(counts)
        total_values = [row.total_ms for row in times]
        embedding_values = [row.embedding_ms for row in times]
        guard_values = [row.guard_ms for row in times]
        summaries.append({
            "split": split,
            "strategy": strategy,
            "threshold": None if threshold == "" else float(threshold),
            **counts,
            **rates,
            "bootstrap_95_ci_by_family": bootstrap_intervals(rows),
            "latency_repeats_per_query": len(times) // len(rows),
            "mean_lookup_ms": mean(total_values),
            "p50_lookup_ms": median(total_values),
            "p95_lookup_ms": _percentile(total_values, 0.95),
            "mean_embedding_ms": mean(embedding_values),
            "p50_embedding_ms": median(embedding_values),
            "mean_guard_ms": mean(guard_values),
            "p95_guard_ms": _percentile(guard_values, 0.95),
            "derived_sequential_lookup_throughput_qps": 1000 / mean(total_values),
        })
    return summaries


def category_breakdown(predictions: list[Prediction]) -> list[dict[str, object]]:
    keys = sorted({(row.split, row.strategy, row.threshold, row.category) for row in predictions})
    output = []
    for split, strategy, threshold, category in keys:
        rows = [row for row in predictions if (row.split, row.strategy, row.threshold, row.category) == (split, strategy, threshold, category)]
        output.append({"split": split, "strategy": strategy, "threshold": None if threshold == "" else float(threshold), "category": category, **_counts(rows), **_rates(_counts(rows))})
    return output


def select_guarded_on_dev(summaries: list[dict[str, object]]) -> float:
    candidates = [row for row in summaries if row["split"] == "dev" and row["strategy"] == "guarded"]
    acceptable = [row for row in candidates if row["unsafe_reuse_rate"] <= 0.02]
    pool = acceptable or candidates
    selected = max(pool, key=lambda row: (row["safe_hit_coverage"], row["correct_hit_precision"] or 0, -row["threshold"]))
    return float(selected["threshold"])


def write_csv(path: Path, rows: list[object]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0])))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def write_charts(summaries: list[dict[str, object]], category_rows: list[dict[str, object]], output: Path) -> None:
    test = [row for row in summaries if row["split"] == "test" and row["strategy"] in {"semantic", "guarded"}]
    for metric, title, filename in (
        ("unsafe_reuse_rate", "Hit rate vs unsafe reuse rate", "01_hit_vs_unsafe_reuse.png"),
        ("correct_hit_precision", "Correct-hit precision vs threshold", "02_precision_vs_threshold.png"),
    ):
        fig, ax = plt.subplots(figsize=(8, 5))
        for strategy in ("semantic", "guarded"):
            rows = sorted((row for row in test if row["strategy"] == strategy), key=lambda row: row["threshold"])
            if metric == "unsafe_reuse_rate":
                ax.plot([row["hit_rate"] for row in rows], [row[metric] for row in rows], marker="o", label=strategy)
                ax.set(xlabel="Hit rate (hits / all held-out queries)", ylabel="Unsafe reuse rate (incorrect hits / all held-out queries)")
                for row in rows:
                    ax.annotate(f"{row['threshold']:.2f}", (row["hit_rate"], row[metric]), fontsize=8)
            else:
                ax.plot([row["threshold"] for row in rows], [row[metric] for row in rows], marker="o", label=strategy)
                ax.set(xlabel="Cosine threshold", ylabel="Correct-hit precision (correct hits / all hits)")
        ax.set_title(title); ax.set_ylim(-0.03, 1.03); ax.grid(alpha=0.25); ax.legend(); fig.tight_layout(); fig.savefig(output / filename, dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    labels, coverage, unsafe = [], [], []
    exact = next(row for row in summaries if row["split"] == "test" and row["strategy"] == "exact")
    chosen = [exact] + sorted(test, key=lambda row: (row["strategy"], row["threshold"]))
    for row in chosen:
        labels.append("exact" if row["strategy"] == "exact" else f"{row['strategy']}\n{row['threshold']:.2f}")
        coverage.append(row["safe_hit_coverage"]); unsafe.append(row["unsafe_reuse_rate"])
    x = range(len(labels)); ax.bar([i - .2 for i in x], coverage, width=.4, label="Safe-hit coverage"); ax.bar([i + .2 for i in x], unsafe, width=.4, label="Unsafe reuse rate")
    ax.set(title="Baseline versus guarded cache outcomes", ylabel="Rate over all held-out queries", xticks=list(x), xticklabels=labels, ylim=(0, 1)); ax.legend(); fig.tight_layout(); fig.savefig(output / "03_strategy_comparison.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5)); width = .35
    ax.bar([i - width/2 for i in x], [row["p50_lookup_ms"] for row in chosen], width, label="p50")
    ax.bar([i + width/2 for i in x], [row["p95_lookup_ms"] for row in chosen], width, label="p95")
    ax.set(title="Warm lookup latency", ylabel="Milliseconds per lookup", xticks=list(x), xticklabels=labels); ax.legend(); fig.tight_layout(); fig.savefig(output / "04_lookup_latency.png", dpi=160); plt.close(fig)

    guarded_085 = [row for row in category_rows if row["split"] == "test" and row["strategy"] == "guarded" and row["threshold"] == 0.85 and row["ineligible_queries"]]
    guarded_085.sort(key=lambda row: row["unsafe_reuse_rate"], reverse=True)
    fig, ax = plt.subplots(figsize=(10, 6)); ax.barh([row["category"] for row in guarded_085], [row["unsafe_reuse_rate"] for row in guarded_085]); ax.invert_yaxis(); ax.set(title="Guarded-cache failures by negative category (threshold 0.85)", xlabel="Incorrect hits / category queries"); fig.tight_layout(); fig.savefig(output / "05_failure_categories.png", dpi=160); plt.close(fig)


METRIC_DEFINITIONS = {
    "hit_rate": "all cache hits / all evaluated queries",
    "correct_hit_precision": "correct hits / all cache hits",
    "incorrect_hit_rate_among_hits": "incorrect hits / all cache hits",
    "unsafe_reuse_rate": "incorrect hits / all evaluated queries",
    "safe_hit_coverage": "correct hits / all evaluated queries",
    "reuse_eligibility_recall": "correct hits / ground-truth reuse-eligible queries",
    "false_rejection_rate": "eligible queries not correctly reused / ground-truth reuse-eligible queries",
    "negative_case_false_hit_rate": "incorrect hits / ground-truth ineligible queries (the V1 'false-hit rate on contrasts')",
}


def environment_info(model: str, model_load_ms: float, warmup_ms: float, repeats: int) -> dict[str, object]:
    packages = {}
    for name in ["sentence-transformers", "torch", "transformers", "numpy", "matplotlib", "pytest"]:
        try: packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: packages[name] = None
    return {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(), "random_seed": RANDOM_SEED,
        "python": platform.python_version(), "platform": platform.platform(), "processor": platform.processor(),
        "embedding_model": model, "device": "cpu", "model_load_ms": model_load_ms,
        "first_warmup_embedding_ms": warmup_ms, "benchmark_mode": "warm model, pairwise fresh one-entry cache",
        "latency_repeats_per_query": repeats, "cache_capacity": 1, "queries_populate_cache": False,
        "llm_inference_included": False, "bootstrap_samples": BOOTSTRAP_SAMPLES, "packages": packages,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/test_cases_v2.csv"))
    parser.add_argument("--output", type=Path, default=Path("results_v2"))
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.70, 0.85, 0.90, 0.95])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cases = read_cases(args.data); args.output.mkdir(parents=True, exist_ok=True)
    model_started = perf_counter_ns(); embedder = SentenceTransformerEmbedder(args.model); model_load_ms = (perf_counter_ns() - model_started) / 1_000_000
    warmup_started = perf_counter_ns(); embedder.encode(["deterministic warmup query"]); warmup_ms = (perf_counter_ns() - warmup_started) / 1_000_000
    predictions: list[Prediction] = []; measurements: list[Measurement] = []
    configurations = [("exact", None)] + [(strategy, threshold) for strategy in ("semantic", "guarded") for threshold in args.thresholds]
    for strategy, threshold in configurations:
        logging.info("running strategy=%s threshold=%s", strategy, threshold)
        new_predictions, new_measurements = run_configuration(cases, strategy, threshold, embedder, args.repeats)
        predictions.extend(new_predictions); measurements.extend(new_measurements)
    summaries = summarize(predictions, measurements); categories = category_breakdown(predictions)
    selected = select_guarded_on_dev(summaries)
    write_csv(args.output / "predictions.csv", predictions); write_csv(args.output / "latency_measurements.csv", measurements)
    payload = {"metric_definitions": METRIC_DEFINITIONS, "selected_guarded_threshold_from_dev": selected, "summaries": summaries}
    (args.output / "summary_metrics.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (args.output / "category_metrics.json").write_text(json.dumps(categories, indent=2) + "\n", encoding="utf-8")
    (args.output / "environment.json").write_text(json.dumps(environment_info(args.model, model_load_ms, warmup_ms, args.repeats), indent=2) + "\n", encoding="utf-8")
    write_charts(summaries, categories, args.output)
    logging.info("wrote %d predictions and %d timing samples; selected guarded threshold %.2f", len(predictions), len(measurements), selected)


if __name__ == "__main__":
    main()
