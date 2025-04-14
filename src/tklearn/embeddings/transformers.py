from __future__ import annotations

from typing import ClassVar

import numpy as np
from numpy.typing import ArrayLike
from sentence_transformers import SentenceTransformer

from tklearn import logging
from tklearn.embeddings.base import Embedding, EmbeddingConfig, TextEncoder

__all__ = [
    "Embedding",
]

logger = logging.get_logger(__name__)


class TransformersEmbeddingConfig(EmbeddingConfig):
    loader: ClassVar[str] = "transformers"
    name: str | None = "all-MiniLM-L6-v2"
    device: str = "auto"
    verbose: bool | int = 1


class TransformerEmbedding(Embedding):
    config: TransformersEmbeddingConfig

    def get_vectors(self) -> dict[str, ArrayLike]:
        """Load resource."""
        return {}

    def get_encoder(self) -> None:
        """Return the model."""
        encoder = SentenceTransformer(
            self.config.name, device=self.config.device
        )
        if self.config.verbose:
            logger.info(
                f"Loading model {self.config.name} on device {self.config.device}"
            )
        return TransformerWrapper(encoder)


class TransformerWrapper(TextEncoder):
    def __init__(self, model: SentenceTransformer):
        self.model = model

    def encode(self, texts: str | list[str]) -> np.ndarray:
        if isinstance(texts, tuple):
            texts = " ".join(texts)
        return self.model.encode(texts, convert_to_numpy=True)
