"""Run the controlled cache experiment and write reproducible artifacts."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import importlib.metadata
import json
import logging
from pathlib import Path
import platform
from statistics import mean, median
from time import perf_counter_ns
from typing import Iterable

import matplotlib.pyplot as plt

from .cache import ExactMatchCache, SemanticCache
from .embedding import SentenceTransformerEmbedder


@dataclass
class Result:
    strategy: str
    threshold: str
    case_id: str
    category: str
    acceptable_reuse: bool
    hit: bool
    correct_reuse: bool
    incorrect_reuse: bool
    similarity: str
    lookup_ms: float
    pipeline_ms: float
    matched_query: str
    query: str


def read_cases(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def run_strategy(cases: Iterable[dict[str, str]], strategy: str, threshold: float | None, embedder=None) -> list[Result]:
    results: list[Result] = []
    for case in cases:
        if strategy == "exact":
            cache = ExactMatchCache()
        else:
            cache = SemanticCache(embedder, threshold=threshold if threshold is not None else 0.85)

        seed_scope = case["seed_scope"]
        query_scope = case["query_scope"]
        seed_user = case["seed_user_id"] or None
        query_user = case["query_user_id"] or None
        cache.put(
            case["seed_query"],
            case["seed_answer"],
            scope=seed_scope,
            user_id=seed_user,
            contains_user_data=case["contains_user_data"].lower() == "true",
        )
        started = perf_counter_ns()
        hit = cache.get(case["query"], scope=query_scope, user_id=query_user)
        pipeline_ms = (perf_counter_ns() - started) / 1_000_000
        acceptable = case["acceptable_reuse"].lower() == "true"
        is_hit = hit is not None
        correct = is_hit and acceptable and hit.response == case["expected_answer"]
        incorrect = is_hit and not correct
        results.append(
            Result(
                strategy=strategy,
                threshold="" if threshold is None else f"{threshold:.2f}",
                case_id=case["case_id"],
                category=case["category"],
                acceptable_reuse=acceptable,
                hit=is_hit,
                correct_reuse=correct,
                incorrect_reuse=incorrect,
                similarity="" if hit is None else f"{hit.similarity:.6f}",
                lookup_ms=pipeline_ms if hit is None else hit.lookup_ns / 1_000_000,
                pipeline_ms=pipeline_ms,
                matched_query="" if hit is None else hit.matched_query,
                query=case["query"],
            )
        )
    return results


def summarize(rows: list[Result]) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    keys = sorted({(row.strategy, row.threshold) for row in rows})
    for strategy, threshold in keys:
        group = [row for row in rows if (row.strategy, row.threshold) == (strategy, threshold)]
        contrasts = [row for row in group if not row.acceptable_reuse]
        hits = sum(row.hit for row in group)
        correct = sum(row.correct_reuse for row in group)
        incorrect = sum(row.incorrect_reuse for row in group)
        latencies = [row.pipeline_ms for row in group]
        summaries.append(
            {
                "strategy": strategy,
                "threshold": None if threshold == "" else float(threshold),
                "queries": len(group),
                "hits": hits,
                "hit_rate": hits / len(group),
                "correct_reuse": correct,
                "incorrect_reuse": incorrect,
                "reuse_precision": correct / hits if hits else None,
                "false_hit_rate_on_contrasts": sum(row.incorrect_reuse for row in contrasts) / len(contrasts),
                "mean_pipeline_ms": mean(latencies),
                "median_pipeline_ms": median(latencies),
                "p95_pipeline_ms": sorted(latencies)[max(0, int(0.95 * len(latencies)) - 1)],
            }
        )
    return summaries


def write_chart(summaries: list[dict[str, object]], path: Path) -> None:
    semantic = [row for row in summaries if row["strategy"] == "semantic"]
    thresholds = [row["threshold"] for row in semantic]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, [row["hit_rate"] for row in semantic], marker="o", label="Overall hit rate")
    ax.plot(thresholds, [row["false_hit_rate_on_contrasts"] for row in semantic], marker="o", label="False-hit rate (contrasts)")
    ax.plot(thresholds, [row["reuse_precision"] or 0 for row in semantic], marker="o", label="Reuse precision")
    ax.set(xlabel="Cosine-similarity threshold", ylabel="Rate", ylim=(-0.03, 1.03), title="Semantic cache efficiency vs. correctness")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def environment_info(model_name: str) -> dict[str, object]:
    packages = {}
    for name in ["sentence-transformers", "torch", "transformers", "numpy", "matplotlib", "pytest"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "embedding_model": model_name,
        "device": "cpu",
        "packages": packages,
        "llm_inference_included": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/test_cases.csv"))
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.70, 0.75, 0.80, 0.85, 0.90, 0.95])
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args.output.mkdir(parents=True, exist_ok=True)
    cases = read_cases(args.data)
    embedder = SentenceTransformerEmbedder(args.model)
    rows = run_strategy(cases, "exact", None)
    for threshold in args.thresholds:
        rows.extend(run_strategy(cases, "semantic", threshold, embedder))

    with (args.output / "raw_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0])))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)
    summaries = summarize(rows)
    (args.output / "summary_metrics.json").write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
    (args.output / "environment.json").write_text(json.dumps(environment_info(args.model), indent=2) + "\n", encoding="utf-8")
    write_chart(summaries, args.output / "threshold_comparison.png")
    logging.info("wrote %d observations and %d summaries to %s", len(rows), len(summaries), args.output)


if __name__ == "__main__":
    main()
