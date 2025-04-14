from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import ClassVar, Dict

import fasttext
import fasttext.util
import numpy as np
from typing_extensions import Self

from tklearn import config
from tklearn.embeddings.base import Embedding, EmbeddingConfig

__all__ = [
    "FastTextEmbeddingConfig",
    "FastTextEmbedding",
]

logger = logging.getLogger(__name__)


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


class FastTextEmbeddingConfig(EmbeddingConfig):
    loader: ClassVar[str] = "fasttext"
    name: str = "cc.en.300.bin"


class FastTextEmbedding(Embedding):
    config: FastTextEmbeddingConfig

    def __post_init__(self):
        self.files_dir = (
            Path(config.assets_dir) / self.config.loader / "loader"
        )
        super().__post_init__()

    def _fetch_embedding(self) -> Self:
        lang_id = self.config.name.split(".")[1]
        with change_dir(self.files_dir):
            fasttext.util.download_model(lang_id, if_exists="ignore")
        return self

    def _read_embedding(self) -> Dict[str, np.ndarray]:
        fn = self.config.name
        model = fasttext.load_model(f"{self.files_dir / fn}")
        vectors = {}
        for term in model.get_words():
            vectors[term] = model.get_word_vector(term)
        return vectors

    def get_vectors(self) -> Dict[str, np.ndarray]:
        return self._fetch_embedding()._read_embedding()

    def get_encoder(self) -> FastTextWrapper:
        model = fasttext.load_model(f"{self.files_dir / self.config.name}")
        return FastTextWrapper(model)


class FastTextWrapper:
    def __init__(self, model: fasttext.FastText._FastText):
        self.model = model

    def encode(self, texts: str | list[str]) -> np.ndarray:
        """Encode the texts."""
        if isinstance(texts, str):
            texts = [texts]
        vectors = []
        for text in texts:
            vectors.append(self.model.get_word_vector(text))
        return np.array(vectors)

    def get_dimension(self) -> int | None:
        """Get the embedding size."""
        return self.model.get_dimension()
