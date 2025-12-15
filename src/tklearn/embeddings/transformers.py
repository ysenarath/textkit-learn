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
        self.batch_size = 32  # type: int
        self.show_progress_bar = False  # type: bool
        self.encoder = None  # type: str | None
        self.convert_to_numpy = True  # type: bool

    def _encode(
        self,
        texts: str | list[str],
        encoder: str | None = None,
        kwargs: dict | None = None,
    ) -> np.ndarray:
        """Encode the texts."""
        if isinstance(texts, str):
            texts = [texts]
        kwargs = kwargs or {}
        if "convert_to_numpy" not in kwargs:
            kwargs["convert_to_numpy"] = self.convert_to_numpy
        if self.show_progress_bar:
            kwargs["show_progress_bar"] = True
        if self.batch_size and "batch_size" not in kwargs:
            kwargs["batch_size"] = self.batch_size
        encoder = encoder or self.encoder
        if encoder == "document":
            return self.model.encode_document(texts, **kwargs)
        if encoder == "query":
            return self.model.encode_query(texts, **kwargs)
        return self.model.encode(texts, **kwargs)

    def encode(self, texts: str | list[str], **kwargs) -> np.ndarray:
        """Encode the texts."""
        return self._encode(texts, encoder=None, kwargs=kwargs)

    def encode_query(self, texts: str | list[str], **kwargs) -> np.ndarray:
        """Encode the texts as queries."""
        return self._encode(texts, encoder="query", kwargs=kwargs)

    def encode_document(self, texts: str | list[str], **kwargs) -> np.ndarray:
        """Encode the texts as documents."""
        return self._encode(texts, encoder="document", kwargs=kwargs)

    def get_dimension(self) -> int:
        """Get the embedding size."""
        return self.model.get_sentence_embedding_dimension()
