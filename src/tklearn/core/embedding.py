from __future__ import annotations

from typing import Generic, List, Literal, Mapping, Optional, TypeVar, overload

import numpy as np
import torch

from tklearn.core.vocab import Vocab

T = TypeVar("T", List[int], np.ndarray, torch.Tensor)


class Embedding(Mapping[str, T], Generic[T]):
    def __init__(
        self,
        weights: Optional[List[List[float]]] = None,
        tokens: Optional[List[str]] = None,
    ):
        super().__init__()
        if weights is None:
            weights = []
        if tokens is None:
            tokens = []
        if len(weights) != len(tokens):
            raise ValueError(
                f"got {len(weights)} weights and {len(tokens)} tokens"
            )
        self._token2index = {token: i for i, token in enumerate(tokens)}
        self._weights = weights
        self._format = None

    @overload
    def set_format(self, value: Literal["np"]) -> Embedding[np.ndarray]: ...

    @overload
    def set_format(
        self, value: Literal["torch", "pt"]
    ) -> Embedding[torch.Tensor]: ...

    def set_format(self, format: Optional[str]) -> Embedding:
        if self._format == format:
            return self
        if format is not None:
            format = str(format).lower()
            if format == "torch":
                format = "pt"
            elif format == "numpy":
                format = "np"
            if format not in ["np", "pt"]:
                raise ValueError(
                    f"got '{format}', expected one of ['np', 'torch', 'pt']"
                )
        copy = Embedding()
        copy._token2index = self._token2index
        copy._weights = self._weights
        copy._format = format
        return copy

    def __getitem__(self, key: str) -> T:
        e = self._weights[self._token2index[key]]
        if self._format == "pt":
            return torch.from_numpy(e)
        elif self._format == "np":
            return np.array(e)
        return e

    def __len__(self) -> int:
        return len(self._token2index)

    def __iter__(self):
        return iter(self._token2index)

    def __contains__(self, key: str) -> bool:
        return key in self._token2index

    def get_weights(self, vocab: Vocab) -> T:
        n, d = len(vocab), len(self._weights[0])
        weights = np.zeros((n, d), dtype=np.float32)
        for index, token in vocab.items():
            if token in self._token2index:
                weights[index] = self[token]
        if self._format == "pt":
            return torch.from_numpy(weights)
        return weights
