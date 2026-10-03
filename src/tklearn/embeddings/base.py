from __future__ import annotations

import abc
import json
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path

import numpy as np

from tklearn import config, logging

__all__ = [
    "Embedding",
    "WordEmbedding",
]

logger = logging.get_logger(__name__)


class Embedding(abc.ABC):
    """Maps text to fixed-size vectors.

    `encode` returns a 1-D vector for a single string and a 2-D array (one
    row per text) for a sequence of strings. Retrieval-style models can embed
    queries and documents differently through `encode_query` and
    `encode_document`; by default both are `encode`.
    """

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


class WordEmbedding(Embedding, Mapping[str, np.ndarray]):
    """Static word vectors with a fixed vocabulary.

    Vectors are cached once under ``config.assets_dir/<loader>/data`` and
    then memory-mapped, so loading is fast and cheap. The embedding is also a
    read-only mapping from word to vector.

    Subclasses set `loader` and implement `load_vectors`; they can implement
    `encode_oov` to embed words outside the vocabulary.

    Parameters
    ----------
    name : str
        Name of the vector set within the loader.
    verbose : bool, default=True
        Show progress while building the cache.
    """

    loader: str

    def __init__(self, name: str, verbose: bool = True) -> None:
        self.name = name
        self.verbose = verbose
        cache_path = self.cache_dir / "data" / f"vectors-{name}.data"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._load(cache_path)
        except FileNotFoundError:
            logger.info(f"Building the vector cache for {name!r}.")
            self._from_dict(self.load_vectors())
            self._dump(cache_path)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name!r})"

    @property
    def cache_dir(self) -> Path:
        return Path(config.assets_dir) / self.loader

    @abc.abstractmethod
    def load_vectors(self) -> Mapping[str, np.ndarray]:
        """Load the full word-to-vector table from the source."""

    def encode_oov(self, word: str) -> np.ndarray:
        """Embed a word outside the vocabulary.

        Raises
        ------
        KeyError
            Always, unless overridden.
        """
        raise KeyError(word)

    @property
    def dim(self) -> int:
        return self.vectors.shape[1]

    @property
    def shape(self) -> tuple[int, int]:
        """(vocabulary size, dimension)."""
        return self.vectors.shape

    def encode(self, texts: str | Sequence[str], **kwargs) -> np.ndarray:
        if isinstance(texts, str):
            return self[texts]
        return np.stack([self[text] for text in texts])

    def __getitem__(self, word: str) -> np.ndarray:
        index = self.word_to_index.get(word)
        if index is None:
            return self.encode_oov(word)
        return self.vectors[index]

    def __contains__(self, word: object) -> bool:
        return word in self.word_to_index

    def __iter__(self) -> Iterator[str]:
        return iter(self.word_to_index)

    def __len__(self) -> int:
        return len(self.word_to_index)

    # --- cache -----------------------------------------------------------

    def _load(self, path: Path) -> None:
        with open(path.with_suffix(".word_to_index.json")) as f:
            self.word_to_index: dict[str, int] = json.load(f)
        filename = path.with_suffix(".vectors.npy")
        # np.load with mmap_mode reads the header and maps the data in place
        self.vectors: np.ndarray = np.load(filename, mmap_mode="r")

    def _from_dict(self, vectors: Mapping[str, np.ndarray]) -> None:
        self.word_to_index = {word: i for i, word in enumerate(vectors)}
        self.vectors = np.asarray(list(vectors.values()), dtype=np.float32)

    def _dump(self, path: Path) -> None:
        np.save(path.with_suffix(".vectors.npy"), self.vectors)
        with open(path.with_suffix(".word_to_index.json"), "w") as f:
            json.dump(self.word_to_index, f)
