import csv
from pathlib import Path

from data.build_dataset_v2 import RANDOM_SEED, build_rows


def test_dataset_has_credible_scale_and_balanced_labels():
    rows = build_rows()
    assert len(rows) >= 200
    assert sum(row["reuse_eligible"] == "true" for row in rows) >= 100
    assert sum(row["reuse_eligible"] == "false" for row in rows) >= 100
    assert RANDOM_SEED == 20261023


def test_dataset_reproducible_and_matches_tracked_csv():
    generated = build_rows()
    path = Path(__file__).parents[1] / "data" / "test_cases_v2.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        tracked = list(csv.DictReader(handle))
    assert generated == tracked


def test_no_family_leakage_between_dev_and_test():
    splits = {}
    for row in build_rows():
        splits.setdefault(row["family_id"], set()).add(row["split"])
    assert all(len(value) == 1 for value in splits.values())


def test_every_label_has_provenance_and_explanation():
    provenance = set()
    for row in build_rows():
        provenance.add(row["provenance"])
        assert row["label_explanation"].strip()
    assert provenance == {
        "manual_curated_v2_materialized_by_script",
        "programmatic_exact_repeat_from_curated_seed",
    }


def test_small_disclosed_exact_repeat_subset():
    rows = build_rows()
    exact = [row for row in rows if row["category"] == "positive_exact_repeat"]
    assert len(exact) == 12
    assert all(row["query"] == row["seed_query"] for row in exact)
