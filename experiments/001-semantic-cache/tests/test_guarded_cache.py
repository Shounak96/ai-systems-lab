import numpy as np
import pytest

from semantic_cache.cache import GuardedSemanticCache


class FakeEmbedder:
    def __init__(self, vectors):
        self.vectors = vectors

    def encode(self, texts):
        return np.asarray([self.vectors[text] for text in texts], dtype=np.float32)


def test_guarded_cache_accepts_safe_paraphrase():
    vectors = {"Enable backups": [1.0, 0.0], "Turn on backups": [0.95, (1 - 0.95**2) ** 0.5]}
    cache = GuardedSemanticCache(FakeEmbedder(vectors), threshold=0.9)
    cache.put("Enable backups", "yes")
    outcome = cache.lookup("Turn on backups")
    assert outcome.hit is not None
    assert outcome.guard_decision.accepted


def test_guarded_cache_rejects_number_change_above_threshold():
    vectors = {"Use 3 replicas": [1.0, 0.0], "Use 5 replicas": [0.99, (1 - 0.99**2) ** 0.5]}
    cache = GuardedSemanticCache(FakeEmbedder(vectors), threshold=0.9)
    cache.put("Use 3 replicas", "three")
    outcome = cache.lookup("Use 5 replicas")
    assert outcome.hit is None
    assert outcome.similarity == pytest.approx(0.99)
    assert "number_mismatch" in outcome.guard_decision.reasons


def test_guarded_threshold_boundary_remains_inclusive():
    vectors = {"cached": [1.0, 0.0], "incoming": [0.8, 0.6]}
    cache = GuardedSemanticCache(FakeEmbedder(vectors), threshold=0.8)
    cache.put("cached", "response")
    assert cache.lookup("incoming").hit is not None


def test_tenant_isolation_precedes_guarding():
    vectors = {"my settings": [1.0, 0.0]}
    cache = GuardedSemanticCache(FakeEmbedder(vectors), threshold=0.8)
    cache.put("my settings", "alice", scope="user", user_id="alice", contains_user_data=True)
    outcome = cache.lookup("my settings", scope="user", user_id="bob")
    assert outcome.hit is None
    assert outcome.candidate_query is None
    assert outcome.guard_decision is None
