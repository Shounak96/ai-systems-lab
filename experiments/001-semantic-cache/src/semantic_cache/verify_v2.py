"""Independently reconcile V2 JSON summaries with raw CSV artifacts."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import mean, median

from .evaluate_v2 import _percentile


def main() -> None:
    root = Path("results_v2")
    with (root / "predictions.csv").open(encoding="utf-8", newline="") as handle:
        predictions = list(csv.DictReader(handle))
    with (root / "latency_measurements.csv").open(encoding="utf-8", newline="") as handle:
        measurements = list(csv.DictReader(handle))
    payload = json.loads((root / "summary_metrics.json").read_text(encoding="utf-8"))

    for summary in payload["summaries"]:
        threshold = "" if summary["threshold"] is None else f"{summary['threshold']:.2f}"
        key = (summary["split"], summary["strategy"], threshold)
        rows = [row for row in predictions if (row["split"], row["strategy"], row["threshold"]) == key]
        timing = [row for row in measurements if (row["split"], row["strategy"], row["threshold"]) == key]
        counts = {
            "queries": len(rows),
            "eligible_queries": sum(row["reuse_eligible"] == "True" for row in rows),
            "ineligible_queries": sum(row["reuse_eligible"] == "False" for row in rows),
            "hits": sum(row["hit"] == "True" for row in rows),
            "correct_hits": sum(row["correct_hit"] == "True" for row in rows),
            "incorrect_hits": sum(row["incorrect_hit"] == "True" for row in rows),
            "false_rejections": sum(row["false_rejection"] == "True" for row in rows),
        }
        for name, value in counts.items():
            assert summary[name] == value, (key, name, summary[name], value)
        total = [float(row["total_ms"]) for row in timing]
        checks = {
            "mean_lookup_ms": mean(total),
            "p50_lookup_ms": median(total),
            "p95_lookup_ms": _percentile(total, 0.95),
            "derived_sequential_lookup_throughput_qps": 1000 / mean(total),
        }
        for name, value in checks.items():
            assert math.isclose(summary[name], value, rel_tol=1e-12), (key, name, summary[name], value)

    charts = [
        "01_hit_vs_unsafe_reuse.png", "02_precision_vs_threshold.png",
        "03_strategy_comparison.png", "04_lookup_latency.png", "05_failure_categories.png",
    ]
    assert all((root / chart).stat().st_size > 0 for chart in charts)
    print(f"verified {len(predictions)} predictions, {len(measurements)} timings, {len(payload['summaries'])} summaries, and {len(charts)} charts")


if __name__ == "__main__":
    main()
