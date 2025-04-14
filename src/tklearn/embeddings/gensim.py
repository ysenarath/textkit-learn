from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import ClassVar, Dict

import gensim.downloader as api
import numpy as np
import tqdm
from gensim.models.keyedvectors import KeyedVectors

from tklearn import config, logging
from tklearn.embeddings.base import Embedding, EmbeddingConfig

__all__ = [
    "GensimEmbedding",
    "GensimEmbeddingConfig",
]

logger = logging.get_logger(__name__)


@contextmanager
def change_dir(path: str | Path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    old_path = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old_path)


class GensimEmbeddingConfig(EmbeddingConfig):
    loader: ClassVar[str] = "gensim"
    name: str = "word2vec-google-news-300"


class GensimEmbedding(Embedding):
    config: GensimEmbeddingConfig

    def __post_init__(self):
        self.files_dir = (
            Path(config.assets_dir) / self.config.loader / "loader"
        )
        super().__post_init__()

    def _fetch_read_embedding(self) -> Dict[str, np.ndarray]:
        model: KeyedVectors
        with change_dir(self.files_dir):
            model = api.load(self.config.name)
        vectors = {}
        for term in tqdm.tqdm(
            model.index_to_key, disable=not self.config.verbose
        ):
            vectors[term] = np.array(model[term])
        return vectors

    def get_vectors(self) -> Dict[str, np.ndarray]:
        return self._fetch_read_embedding()

    def get_encoder(self) -> GensimModelWrapper:
        """Returns the encoder for the embedding."""
        model: KeyedVectors
        with change_dir(self.files_dir):
            model = api.load(self.config.name)
        return GensimModelWrapper(model)


class GensimModelWrapper:
    def __init__(self, model: KeyedVectors):
        self.model = model

    def encode(self, texts: str | list[str]) -> np.ndarray:
        """Encode a given text string or token span into a vector.

        Parameters
        ----------
        text : str or list[str]
            The text to be encoded. Can be a single string or a list of
            strings (e.g., tokens or phrases).

        Returns
        -------
        np.ndarray
            The numerical vector representation (embedding) of the input text.
        """
        if isinstance(texts, str):
            texts = [texts]
        vectors = []
        for text in texts:
            vectors.append(self.model[text])
        return np.array(vectors)

    def get_dimension(self) -> int | None:
        """Get the size of the embedding vector.

        Returns
        -------
        int
            The dimensionality of the embedding vectors produced by this model.
        """
        return self.model.vector_size
