from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import ClassVar, Dict

import gensim.downloader as api
import numpy as np
import tqdm
from gensim.models.keyedvectors import KeyedVectors

from tklearn import config
from tklearn.embeddings.base import Embedding, EmbeddingConfig

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


class GensimEmbeddingConfig(EmbeddingConfig):
    identifier: ClassVar[str] = "gensim"
    version: str = "word2vec-google-news-300"


class GensimEmbedding(Embedding):
    config: GensimEmbeddingConfig

    def __post_init__(self):
        self.files_dir = (
            Path(config.resources_dir) / self.config.identifier / "loader"
        )
        super().__post_init__()

    def _fetch_read_embedding(self) -> Dict[str, np.ndarray]:
        model: KeyedVectors
        with change_dir(self.files_dir):
            model = api.load(self.config.version)
        vectors = {}
        for term in tqdm.tqdm(
            model.index_to_key, disable=not self.config.verbose
        ):
            vectors[term] = np.array(model[term])
        return vectors

    def load(self) -> Dict[str, np.ndarray]:
        return self._fetch_read_embedding()

    def get_model(self) -> None:
        return None
