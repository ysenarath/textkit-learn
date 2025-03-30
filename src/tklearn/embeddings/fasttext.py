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

logger = logging.getLogger(__name__)


URLS = {"fasttext-cc.en.300": None}


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
    name: ClassVar[str] = "fasttext"
    version: str = "cc.en.300.bin"


class FastTextEmbedding(Embedding):
    config: FastTextEmbeddingConfig

    def __post_init__(self):
        self.files_dir = (
            Path(config.resources_dir) / self.config.name / "loader"
        )
        super().__post_init__()

    def _fetch_embedding(self) -> Self:
        lang_id = self.config.version.split(".")[1]
        with change_dir(self.files_dir):
            fasttext.util.download_model(lang_id, if_exists="ignore")
        return self

    def _read_embedding(self) -> Dict[str, np.ndarray]:
        fn = self.config.version
        model = fasttext.load_model(f"{self.files_dir / fn}")
        vectors = {}
        for term in model.get_words():
            vectors[term] = model.get_word_vector(term)
        return vectors

    def load(self) -> Dict[str, np.ndarray]:
        return self._fetch_embedding()._read_embedding()

    def get_model(self) -> fasttext.FastText:
        fn = self.config.version
        return fasttext.load_model(f"{self.files_dir / fn}")
