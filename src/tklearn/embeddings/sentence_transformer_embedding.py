from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from tklearn.embeddings.base import Embedding

__all__ = [
    "SentenceTransformerEmbedding",
]


class SentenceTransformerEmbedding(Embedding):
    """Sentence embeddings from a ``sentence-transformers`` model.

    Parameters
    ----------
    name : str
        Model id or path for ``SentenceTransformer``, e.g.
        ``"all-mpnet-base-v2"``.
    device : str, optional
        Device for the model; by default CUDA, then MPS, then CPU.
    batch_size : int, default=32
        Default batch size for encoding.
    show_progress_bar : bool, default=False
        Show a progress bar while encoding.

    Notes
    -----
    Extra keyword arguments to the ``encode*`` methods are passed to
    ``SentenceTransformer.encode`` (e.g. ``normalize_embeddings=True``).
    """

    def __init__(
        self,
        name: str,
        device: str | None = None,
        batch_size: int = 32,
        show_progress_bar: bool = False,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self.name = name
        self.batch_size = batch_size
        self.show_progress_bar = show_progress_bar
        self.model = SentenceTransformer(name, device=device)

    @property
    def dim(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def _encode(
        self, method: str, texts: str | Sequence[str], kwargs: dict
    ) -> np.ndarray:
        kwargs.setdefault("batch_size", self.batch_size)
        kwargs.setdefault("show_progress_bar", self.show_progress_bar)
        kwargs.setdefault("convert_to_numpy", True)
        return getattr(self.model, method)(texts, **kwargs)

    def encode(self, texts: str | Sequence[str], **kwargs) -> np.ndarray:
        return self._encode("encode", texts, kwargs)

    def encode_query(self, texts: str | Sequence[str], **kwargs) -> np.ndarray:
        return self._encode("encode_query", texts, kwargs)

    def encode_document(
        self, texts: str | Sequence[str], **kwargs
    ) -> np.ndarray:
        return self._encode("encode_document", texts, kwargs)
