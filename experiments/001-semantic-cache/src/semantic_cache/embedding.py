"""Embedding interfaces and the experiment's sentence-transformer adapter."""

from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np


class EmbeddingModel(Protocol):
    """Small interface that keeps cache logic independent of model loading."""

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Return one L2-normalized vector per input text."""


class SentenceTransformerEmbedder:
    """Local embeddings backed by sentence-transformers/all-MiniLM-L6-v2."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.device = device
        self._model = SentenceTransformer(model_name, device=device)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vectors = self._model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)
