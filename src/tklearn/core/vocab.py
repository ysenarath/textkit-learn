from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Protocol, Tuple, TypeVar, runtime_checkable

import numpy as np
import torch

from tklearn.core.vocab_store import SQLAlchemyVocabStore

T = TypeVar("T", List[int], np.ndarray, torch.Tensor)


@runtime_checkable
class NodeLike(Protocol):
    id: str
    label: str


class Vocab:
    def __init__(self, path: str | Path):
        """Initialize the triplet store with a path to the LevelDB database."""
        self.path = path

    @property
    def path(self) -> Path:
        return self._path

    @path.setter
    def path(self, value: str | Path):
        self._path = Path(value).absolute()
        self.store = None
        self.open()

    def open(self):
        """Open the database connection."""
        url = f"sqlite:///{self.path}.db"
        self.store = SQLAlchemyVocabStore(url)
        return self

    def get_index(self, text: str) -> int:
        try:
            default = self.get_index("[UNK]")
            return self.store.get_token(text, default=default)
        except IndexError:
            return self.store.get_token(text)

    def get_token(self, index: int) -> str:
        return self.store.get_token(index)

    def add(self, item: str | NodeLike) -> Vocab:
        if isinstance(item, NodeLike):
            item = item.label
        self.store.add(item)
        return self

    def update(self, items: Iterable[str | NodeLike]) -> Vocab:
        for item in items:
            self.add(item)
        return self

    def items(self) -> Iterable[Tuple[int, str]]:
        return self.store.items()
