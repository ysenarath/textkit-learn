from __future__ import annotations

import numpy as np
from tqdm import auto as tqdm

from tklearn.embeddings._utils import change_dir
from tklearn.embeddings.base import WordEmbedding

__all__ = [
    "GensimEmbedding",
]


class GensimEmbedding(WordEmbedding):
    """Word vectors from the gensim downloader.

    Parameters
    ----------
    name : str, default="word2vec-google-news-300"
        A ``gensim.downloader`` model name, e.g. ``"glove-wiki-gigaword-100"``.
    verbose : bool, default=True
        Show progress while building the cache.
    """

    loader = "gensim"

    def __init__(
        self, name: str = "word2vec-google-news-300", verbose: bool = True
    ) -> None:
        super().__init__(name, verbose=verbose)

    def load_vectors(self) -> dict[str, np.ndarray]:
        import gensim.downloader as api

        with change_dir(self.cache_dir / "loader"):
            model = api.load(self.name)
        return {
            term: np.asarray(model[term])
            for term in tqdm.tqdm(model.index_to_key, disable=not self.verbose)
        }
