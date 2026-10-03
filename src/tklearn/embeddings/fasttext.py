from __future__ import annotations

import numpy as np
from tqdm import auto as tqdm

from tklearn.embeddings._utils import change_dir
from tklearn.embeddings.base import WordEmbedding

__all__ = [
    "FastTextEmbedding",
]


class FastTextEmbedding(WordEmbedding):
    """fastText word vectors, with subword vectors for unknown words.

    Parameters
    ----------
    name : str, default="cc.en.300.bin"
        A pretrained Common Crawl model, ``"cc.<lang>.300.bin"``.
    verbose : bool, default=True
        Show progress while building the cache.

    Notes
    -----
    Known words are read from the memory-mapped cache. The full fastText
    model is only loaded the first time an out-of-vocabulary word is
    encoded.
    """

    loader = "fasttext"

    def __init__(
        self, name: str = "cc.en.300.bin", verbose: bool = True
    ) -> None:
        self._model = None
        super().__init__(name, verbose=verbose)

    def _load_model(self):
        import fasttext
        import fasttext.util

        files_dir = self.cache_dir / "loader"
        lang_id = self.name.split(".")[1]
        with change_dir(files_dir):
            fasttext.util.download_model(lang_id, if_exists="ignore")
        return fasttext.load_model(str(files_dir / self.name))

    @property
    def model(self):
        if self._model is None:
            self._model = self._load_model()
        return self._model

    def load_vectors(self) -> dict[str, np.ndarray]:
        model = self.model
        return {
            term: model.get_word_vector(term)
            for term in tqdm.tqdm(model.get_words(), disable=not self.verbose)
        }

    def encode_oov(self, word: str) -> np.ndarray:
        return self.model.get_word_vector(word)
