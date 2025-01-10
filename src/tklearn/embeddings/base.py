from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar, Dict, Iterable, Mapping, Optional, Union

import numpy as np
from fasttext.FastText import _FastText as FastText
from nightjar import AutoModule, BaseConfig, BaseModule

from tklearn import config

__all__ = [
    "EmbeddingConfig",
    "AutoEmbedding",
    "Embedding",
]


class EmbeddingConfig(BaseConfig, dispatch="identifier"):
    identifier: ClassVar[str]
    version: str = "0.0.1"
    verbose: Union[bool, int] = 1


class AutoEmbedding(AutoModule):
    def __new__(cls, config: EmbeddingConfig) -> Embedding:
        return super().__new__(cls, config)

    @classmethod
    def from_config(cls, config: EmbeddingConfig | Mapping) -> Embedding:
        if not isinstance(config, EmbeddingConfig):
            try:
                config = EmbeddingConfig.from_dict(config)
            except KeyError:
                config = {
                    "identifier": "gensim",
                    # make the identidier as the version for loading
                    "version": config["identifier"],
                }
                config = EmbeddingConfig.from_dict(config)
        return cls(config)


class Embedding(BaseModule, Mapping[str, np.ndarray]):
    config: EmbeddingConfig
    word_to_index: Optional[Dict[str, int]] = None
    vectors: Optional[np.ndarray] = None
    model: Optional[FastText] = None

    def __post_init__(self) -> None:
        cache_path = (
            Path(config.resources_dir)
            / self.config.identifier
            / "data"
            / f"vectors-{self.config.version}.data"
        )
        # create embedding if not exists
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._load(cache_path)
        except FileNotFoundError:
            mapping = self.load()
            self._from_dict(mapping)
            self._dump(cache_path)
        try:
            self.model = self.get_model()
        except NotImplementedError:
            self.model = None

    def load(self) -> Dict[str, Any]:
        """Load resource."""
        raise NotImplementedError

    def get_model(self) -> Any:
        """Return the model."""
        raise NotImplementedError

    def _load(self, path: Path | str) -> Embedding:
        path = Path(path)
        with open(path.with_suffix(".word_to_index.json")) as f:
            word_to_index = json.load(f)
        vectors: np.ndarray = np.load(path.with_suffix(".vectors.npy"))
        # memory-mapped array
        vectors = np.memmap(
            path.with_suffix(".vectors.npy"),
            dtype=np.float32,
            mode="r",
            shape=vectors.shape,
        )
        self.word_to_index = word_to_index
        self.vectors = vectors

    def _from_dict(self, wv: Dict[str, np.ndarray]) -> Embedding:
        word_to_index = {entity: i for i, entity in enumerate(wv.keys())}
        vectors = np.array(list(wv.values()))
        self.word_to_index = word_to_index
        self.vectors = vectors

    def _dump(self, path: Path | str) -> None:
        path = Path(path)
        np.save(path.with_suffix(".vectors.npy"), self.vectors)
        with open(path.with_suffix(".word_to_index.json"), "w") as f:
            json.dump(self.word_to_index, f)

    def get_word_vector(self, word: str) -> np.ndarray:
        if self.model:
            return self.model.get_word_vector(word)
        return self[word]

    def __getitem__(self, key: str) -> np.ndarray:
        return self.vectors[self.word_to_index[key]]

    def __iter__(self) -> Iterable[str]:
        return iter(self.word_to_index)

    def __len__(self) -> int:
        return len(self.word_to_index)

    @property
    def shape(self) -> tuple[int, int]:
        return self.vectors.shape
