from __future__ import annotations

from typing import ClassVar, Union

import numpy as np
from numpy.typing import ArrayLike
from sentence_transformers import SentenceTransformer

from tklearn import logging
from tklearn.embeddings.base import Embedding, EmbeddingConfig
from tklearn.nn.utils import get_device

__all__ = [
    "SentenceTransformerEmbeddingConfig",
    "SentenceTransformerEmbedding",
]

logger = logging.get_logger(__name__)


class SentenceTransformerEmbeddingConfig(EmbeddingConfig):
    loader: ClassVar[str] = "transformers"
    name: str = "all-MiniLM-L6-v2"
    device: str = "auto"
    verbose: Union[bool, int] = 1


class SentenceTransformerEmbedding(Embedding):
    config: SentenceTransformerEmbeddingConfig

    def get_vectors(self) -> dict[str, ArrayLike]:
        """Load resource."""
        return {}

    def get_encoder(self) -> TransformerWrapper:
        """Return the model."""
        device = get_device(self.config.device)
        encoder = SentenceTransformer(self.config.name, device=device)
        return TransformerWrapper(encoder)


class TransformerWrapper:
    def __init__(self, model: SentenceTransformer):
        self.model = model

    def encode(
        self, texts: str | list[str], batch_size: int = 32, **kwargs
    ) -> np.ndarray:
        """Encode the texts."""
        if isinstance(texts, str):
            texts = [texts]
        return self.model.encode(
            texts, convert_to_numpy=True, batch_size=batch_size
        )

    def get_dimension(self) -> int:
        """Get the embedding size."""
        return self.model.get_sentence_embedding_dimension()
