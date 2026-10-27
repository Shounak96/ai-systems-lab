import pytest

from semantic_cache.evaluate_v2 import Measurement, Prediction, _counts, _rates, summarize


def prediction(case, eligible, hit, correct, incorrect):
    return Prediction("test", "guarded", "0.85", case, case.split("-")[0], "category", eligible, hit, correct, incorrect, eligible and not correct, "0.9", "seed", "True", "", "query", "seed")


def test_metric_denominators_are_explicit_and_correct():
    rows = [
        prediction("a-1", True, True, True, False),
        prediction("b-1", True, False, False, False),
        prediction("c-1", False, True, False, True),
        prediction("d-1", False, False, False, False),
    ]
    counts = _counts(rows)
    rates = _rates(counts)
    assert rates["hit_rate"] == 0.5
    assert rates["correct_hit_precision"] == 0.5
    assert rates["incorrect_hit_rate_among_hits"] == 0.5
    assert rates["unsafe_reuse_rate"] == 0.25
    assert rates["safe_hit_coverage"] == 0.25
    assert rates["reuse_eligibility_recall"] == 0.5
    assert rates["negative_case_false_hit_rate"] == 0.5


def test_summary_latency_and_throughput_calculation():
    rows = [prediction("a-1", True, True, True, False)]
    measurements = [
        Measurement("test", "guarded", "0.85", "a-1", "a", 1, 2.0, 1.5, 0.4, 0.1),
        Measurement("test", "guarded", "0.85", "a-1", "a", 2, 4.0, 3.5, 0.4, 0.1),
    ]
    result = summarize(rows, measurements)[0]
    assert result["mean_lookup_ms"] == 3.0
    assert result["p50_lookup_ms"] == 3.0
    assert result["derived_sequential_lookup_throughput_qps"] == pytest.approx(1000 / 3)
