"""In-memory exact-match and cosine-similarity response caches."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import logging
import re
from time import perf_counter_ns
from typing import Literal

import numpy as np

from .embedding import EmbeddingModel
from .guards import GuardConfig, GuardDecision, compare_sensitive_attributes

Scope = Literal["global", "user"]
LOGGER = logging.getLogger(__name__)


def normalize_query(query: str) -> str:
    """Apply intentionally conservative normalization for exact matching."""
    return re.sub(r"\s+", " ", query.strip()).casefold()


def _partition(scope: Scope, user_id: str | None) -> str:
    if scope == "global":
        return "global"
    if scope == "user":
        if not user_id:
            raise ValueError("user_id is required for user-scoped cache access")
        return f"user:{user_id}"
    raise ValueError(f"unsupported cache scope: {scope}")


def _validate_privacy(scope: Scope, contains_user_data: bool, allow_private_global: bool) -> None:
    if scope == "global" and contains_user_data and not allow_private_global:
        raise ValueError("user-specific data cannot be stored in global scope")


@dataclass(frozen=True)
class CacheHit:
    response: str
    matched_query: str
    similarity: float
    lookup_ns: int


@dataclass(frozen=True)
class LookupOutcome:
    """Full lookup decision, including misses and timing components."""

    hit: CacheHit | None
    candidate_query: str | None
    similarity: float | None
    embedding_ns: int
    search_ns: int
    guard_ns: int = 0
    guard_decision: GuardDecision | None = None

    @property
    def total_ns(self) -> int:
        return self.embedding_ns + self.search_ns + self.guard_ns


@dataclass
class _Entry:
    query: str
    response: str
    vector: np.ndarray | None = None


class ExactMatchCache:
    """Normalized exact-match cache partitioned by global or user scope."""

    def __init__(self, max_size: int | None = None, allow_private_global: bool = False) -> None:
        if max_size is not None and max_size < 1:
            raise ValueError("max_size must be positive")
        self.max_size = max_size
        self.allow_private_global = allow_private_global
        self._entries: OrderedDict[tuple[str, str], _Entry] = OrderedDict()

    def put(
        self,
        query: str,
        response: str,
        *,
        scope: Scope = "global",
        user_id: str | None = None,
        contains_user_data: bool = False,
    ) -> None:
        _validate_privacy(scope, contains_user_data, self.allow_private_global)
        key = (_partition(scope, user_id), normalize_query(query))
        self._entries[key] = _Entry(query=query, response=response)
        self._entries.move_to_end(key)
        self._evict()

    def get(
        self, query: str, *, scope: Scope = "global", user_id: str | None = None
    ) -> CacheHit | None:
        started = perf_counter_ns()
        key = (_partition(scope, user_id), normalize_query(query))
        entry = self._entries.get(key)
        elapsed = perf_counter_ns() - started
        if entry is None:
            LOGGER.debug("exact cache miss partition=%s", key[0])
            return None
        self._entries.move_to_end(key)
        LOGGER.debug("exact cache hit partition=%s", key[0])
        return CacheHit(entry.response, entry.query, 1.0, elapsed)

    def _evict(self) -> None:
        if self.max_size is not None:
            while len(self._entries) > self.max_size:
                self._entries.popitem(last=False)


class SemanticCache:
    """Linear-scan cosine cache suitable for a small controlled experiment."""

    def __init__(
        self,
        embedder: EmbeddingModel,
        threshold: float = 0.85,
        max_size: int | None = None,
        allow_private_global: bool = False,
    ) -> None:
        if not -1.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between -1 and 1")
        if max_size is not None and max_size < 1:
            raise ValueError("max_size must be positive")
        self.embedder = embedder
        self.threshold = threshold
        self.max_size = max_size
        self.allow_private_global = allow_private_global
        self._entries: OrderedDict[tuple[str, str], _Entry] = OrderedDict()

    def put(
        self,
        query: str,
        response: str,
        *,
        scope: Scope = "global",
        user_id: str | None = None,
        contains_user_data: bool = False,
    ) -> None:
        _validate_privacy(scope, contains_user_data, self.allow_private_global)
        partition = _partition(scope, user_id)
        vector = self.embedder.encode([query])[0]
        key = (partition, normalize_query(query))
        self._entries[key] = _Entry(query=query, response=response, vector=vector)
        self._entries.move_to_end(key)
        self._evict()

    def get(
        self, query: str, *, scope: Scope = "global", user_id: str | None = None
    ) -> CacheHit | None:
        return self.lookup(query, scope=scope, user_id=user_id).hit

    def lookup(
        self, query: str, *, scope: Scope = "global", user_id: str | None = None
    ) -> LookupOutcome:
        partition = _partition(scope, user_id)
        candidates = [(key, entry) for key, entry in self._entries.items() if key[0] == partition]
        embedding_started = perf_counter_ns()
        query_vector = self.embedder.encode([query])[0]
        embedding_ns = perf_counter_ns() - embedding_started
        search_started = perf_counter_ns()
        if not candidates:
            LOGGER.debug("semantic cache miss (empty partition) partition=%s", partition)
            return LookupOutcome(None, None, None, embedding_ns, perf_counter_ns() - search_started)
        similarities = [float(np.dot(query_vector, entry.vector)) for _, entry in candidates]
        best_index = int(np.argmax(similarities))
        best_similarity = similarities[best_index]
        key, entry = candidates[best_index]
        search_ns = perf_counter_ns() - search_started
        if best_similarity < self.threshold:
            LOGGER.debug("semantic cache miss similarity=%.4f", best_similarity)
            return LookupOutcome(None, entry.query, best_similarity, embedding_ns, search_ns)
        self._entries.move_to_end(key)
        LOGGER.debug("semantic cache hit similarity=%.4f", best_similarity)
        hit = CacheHit(entry.response, entry.query, best_similarity, embedding_ns + search_ns)
        return LookupOutcome(hit, entry.query, best_similarity, embedding_ns, search_ns)

    def _evict(self) -> None:
        if self.max_size is not None:
            while len(self._entries) > self.max_size:
                self._entries.popitem(last=False)


class GuardedSemanticCache(SemanticCache):
    """Semantic cache requiring configurable query-attribute guards to pass."""

    def __init__(self, *args, guard_config: GuardConfig = GuardConfig(), **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.guard_config = guard_config

    def lookup(
        self, query: str, *, scope: Scope = "global", user_id: str | None = None
    ) -> LookupOutcome:
        outcome = super().lookup(query, scope=scope, user_id=user_id)
        if outcome.hit is None:
            return outcome
        guard_started = perf_counter_ns()
        decision = compare_sensitive_attributes(query, outcome.hit.matched_query, self.guard_config)
        guard_ns = perf_counter_ns() - guard_started
        if not decision.accepted:
            LOGGER.debug("guarded cache rejection reasons=%s", decision.reasons)
            return LookupOutcome(
                None,
                outcome.candidate_query,
                outcome.similarity,
                outcome.embedding_ns,
                outcome.search_ns,
                guard_ns,
                decision,
            )
        hit = CacheHit(
            outcome.hit.response,
            outcome.hit.matched_query,
            outcome.hit.similarity,
            outcome.total_ns + guard_ns,
        )
        return LookupOutcome(
            hit,
            outcome.candidate_query,
            outcome.similarity,
            outcome.embedding_ns,
            outcome.search_ns,
            guard_ns,
            decision,
        )
