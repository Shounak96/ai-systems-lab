from __future__ import annotations

import numpy as np
import pytest

from semantic_cache.cache import ExactMatchCache, SemanticCache, normalize_query


class FakeEmbedder:
    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors

    def encode(self, texts):
        return np.asarray([self.vectors[text] for text in texts], dtype=np.float32)


def test_normalization_collapses_case_and_whitespace():
    assert normalize_query("  Hello\n WORLD ") == "hello world"


def test_exact_hit_and_miss():
    cache = ExactMatchCache()
    cache.put("Hello world", "answer")
    assert cache.get(" HELLO   world ").response == "answer"
    assert cache.get("Hello worlds") is None


def test_semantic_threshold_is_inclusive():
    embedder = FakeEmbedder({"seed": [1.0, 0.0], "boundary": [0.8, 0.6]})
    cache = SemanticCache(embedder, threshold=0.8)
    cache.put("seed", "answer")
    hit = cache.get("boundary")
    assert hit is not None
    assert hit.similarity == pytest.approx(0.8)


def test_similar_looking_contrast_can_be_a_false_hit():
    embedder = FakeEmbedder({"12 times 8": [1.0, 0.0], "13 times 8": [0.99, 0.01]})
    cache = SemanticCache(embedder, threshold=0.9)
    cache.put("12 times 8", "96")
    assert cache.get("13 times 8").response == "96"
    assert cache.get("13 times 8").response != "104"


def test_user_scope_isolation_for_exact_and_semantic():
    exact = ExactMatchCache()
    exact.put("my balance", "$10", scope="user", user_id="alice", contains_user_data=True)
    assert exact.get("my balance", scope="user", user_id="bob") is None

    embedder = FakeEmbedder({"my balance": [1.0, 0.0]})
    semantic = SemanticCache(embedder, threshold=0.8)
    semantic.put("my balance", "$10", scope="user", user_id="alice", contains_user_data=True)
    assert semantic.get("my balance", scope="user", user_id="bob") is None


def test_private_data_rejected_in_global_scope():
    with pytest.raises(ValueError, match="user-specific"):
        ExactMatchCache().put("my balance", "$10", contains_user_data=True)


def test_lru_eviction():
    cache = ExactMatchCache(max_size=2)
    cache.put("one", "1")
    cache.put("two", "2")
    assert cache.get("one") is not None
    cache.put("three", "3")
    assert cache.get("two") is None
    assert cache.get("one") is not None
