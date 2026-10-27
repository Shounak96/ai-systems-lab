from semantic_cache.evaluate import Result, summarize


def test_summary_uses_labels_not_similarity_for_correctness():
    rows = [
        Result("semantic", "0.80", "a", "paraphrase", True, True, True, False, "0.9", 1, 1, "q", "p"),
        Result("semantic", "0.80", "b", "negation", False, True, False, True, "0.99", 1, 1, "q", "not q"),
    ]
    summary = summarize(rows)[0]
    assert summary["hit_rate"] == 1.0
    assert summary["reuse_precision"] == 0.5
    assert summary["false_hit_rate_on_contrasts"] == 1.0
