from semantic_cache.guards import GuardConfig, compare_sensitive_attributes


def reasons(incoming, cached, config=GuardConfig()):
    return set(compare_sensitive_attributes(incoming, cached, config).reasons)


def test_number_mismatch_and_equivalent_formatting():
    assert "number_mismatch" in reasons("use 200 requests", "use 100 requests")
    assert "number_mismatch" not in reasons("use 1,000 requests", "use 1000 requests")
    assert "number_mismatch" not in reasons("Use AES-256.", "Use AES-256 before upload.")


def test_date_and_year_mismatch_regression():
    decision = compare_sensitive_attributes(
        "How many days are in February 2023?",
        "How many days are in February 2024?",
    )
    assert not decision.accepted
    assert "date_mismatch" in decision.reasons


def test_backup_negation_regression():
    decision = compare_sensitive_attributes(
        "Should automatic backups not be enabled?",
        "Should automatic backups be enabled?",
    )
    assert not decision.accepted
    assert {"polarity_mismatch", "negation_mismatch"} & set(decision.reasons)


def test_version_mismatch_and_equivalent_version_format():
    assert "version_mismatch" in reasons("Use Python 3.11", "Use Python 3.12")
    assert "version_mismatch" not in reasons("Use API v2.0", "Use API version 2.0")
    assert "version_mismatch" not in reasons("Use API v2", "Use API version 2")
    assert "number_mismatch" not in reasons("Use API v2", "Use API version 2")
    assert "version_mismatch" not in reasons("Use PostgreSQL 15", "Use PostgreSQL v15")
    assert "version_mismatch" not in reasons("Use API 2.0", "Use API version 2")


def test_entity_guard_detects_known_product_change():
    assert "entity_mismatch" in reasons("Use MySQL 8.0", "Use PostgreSQL 8.0")
    assert "entity_mismatch" not in reasons("Use an S3 bucket", "Use an Amazon S3 bucket")


def test_no_and_without_have_same_negative_polarity():
    assert "negation_mismatch" not in reasons("Migrate with no downtime", "Migrate without downtime")
    assert "polarity_mismatch" not in reasons("Run on PostgreSQL with no downtime", "Use PostgreSQL without downtime")


def test_guards_are_independently_configurable():
    config = GuardConfig(numbers=False, dates=False, negation=False, versions=False, entities=False)
    assert compare_sensitive_attributes("Never use Python 3.11", "Use Redis 7.0", config).accepted


def test_correctly_accepts_unmarked_paraphrase():
    assert compare_sensitive_attributes(
        "Configure a timeout for upstream calls.",
        "Set the upstream request timeout.",
    ).accepted


def test_conservative_number_guard_can_false_reject_number_words():
    decision = compare_sensitive_attributes("Use three retries", "Use 3 retries")
    assert not decision.accepted
    assert "number_mismatch" in decision.reasons
