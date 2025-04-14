from __future__ import annotations

from typing import ClassVar

from numpy.typing import ArrayLike
from sentence_transformers import SentenceTransformer

from tklearn import logging
from tklearn.embeddings.base import Embedding, EmbeddingConfig

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
        self.model = SentenceTransformer(
            self.config.name, device=self.config.device
        )
        if self.config.verbose:
            logger.info(
                f"Loading model {self.config.name} on device {self.config.device}"
            )
        return self.model
