from __future__ import annotations

import abc
from collections.abc import Sequence

import numpy as np

__all__ = [
    "Embedding",
]


class Embedding(abc.ABC):
    """Maps text to fixed-size vectors.

    `encode` returns a 1-D vector for a single string and a 2-D array (one
    row per text) for a sequence of strings. Retrieval-style models can embed
    queries and documents differently through `encode_query` and
    `encode_document`; by default both are `encode`.

    Attributes
    ----------
    name : str
        The model, e.g. ``"all-mpnet-base-v2"``.
    """

    name: str

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name!r})"

    @property
    @abc.abstractmethod
    def dim(self) -> int:
        """Size of the vectors."""

    @abc.abstractmethod
    def encode(self, texts: str | Sequence[str], **kwargs) -> np.ndarray:
        """Embed one text (1-D result) or several texts (2-D result)."""

    def encode_query(self, texts: str | Sequence[str], **kwargs) -> np.ndarray:
        """Embed texts used as search queries."""
        return self.encode(texts, **kwargs)

    def encode_document(
        self, texts: str | Sequence[str], **kwargs
    ) -> np.ndarray:
        """Embed texts used as searchable documents."""
        return self.encode(texts, **kwargs)
