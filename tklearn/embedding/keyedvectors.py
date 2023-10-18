from collections.abc import Iterator, Mapping, Sequence
from typing import Union, List, Tuple

import numpy as np
import hnswlib

__all__ = [
    "KeyedVectors",
]


class KeyedVectors(Mapping):
    def __init__(
        self,
        data: Union[np.ndarray, Mapping],
        index_to_key: List[str] = None,
    ):
        # prepare
        if isinstance(data, np.ndarray):
            if index_to_key is None:
                raise ValueError("index_to_key arg is None")
        elif isinstance(data, Mapping):
            arr: List = []
            index_to_key = []
            for key, value in data.items():
                arr.append(value)
                index_to_key.append(key)
            data = np.array(arr)
        elif hasattr(data, "vocab"):
            index_to_key = data.vocab
            data = np.array([data[key] for key in index_to_key])
        elif hasattr(data, "index_to_key"):
            index_to_key = data.index_to_key
            data = np.array([data[key] for key in index_to_key])
        # init attributes
        self.index_to_key = index_to_key
        self.data = data
        # derived attributes
        self.key_to_index = {key: i for i, key in enumerate(self.index_to_key)}
        # create index
        self._index = None

    def insert(self, index: int, key: str, value: np.ndarray = None):
        index_to_key = self.index_to_key.copy()
        index_to_key.insert(index, key)
        if value is None:
            value = np.zeros_like(self.data[0])
        data = np.insert(
            self.data,
            index,
            value,
            axis=0,
        )
        return KeyedVectors(data, index_to_key=index_to_key)

    def append(self, key, value):
        index_to_key = self.index_to_key.copy()
        index_to_key.append(key)
        if value is None:
            value = np.zeros_like(self.data[0])
        data = np.append(
            self.data,
            value,
            axis=0,
        )
        return KeyedVectors(data, index_to_key=index_to_key)

    @property
    def vectors(self) -> np.ndarray:
        return self.data

    def get_vector(self, key: str, norm: bool = False) -> np.ndarray:
        vec = self.data[self.key_to_index[key]]
        if norm:
            vec = vec / np.linalg.norm(vec)
        return vec

    def __getitem__(self, key: Union[str, List[str], int, List[int]]) -> np.ndarray:
        if isinstance(key, str):
            key = self.key_to_index[key]
        elif isinstance(key, Sequence) and len(key) > 0 and isinstance(key[0], str):
            key = [self.key_to_index[k] for k in key]
        return self.data[key]

    def __iter__(self) -> Iterator:
        yield from self.index_to_key

    def __len__(self) -> int:
        return len(self.data)

    @property
    def shape(self):
        return self.data.shape

    def get_mean_vector(
        self,
        keys: Union[List[Union[str, int, np.ndarray]], np.ndarray],
        weights: Union[List[float], np.ndarray] = None,
        pre_normalize: bool = True,
        post_normalize: bool = False,
        ignore_missing: bool = True,
    ):
        if isinstance(keys, np.ndarray):
            matrix = keys
        else:
            matrix = []
            for key in keys:
                if isinstance(key, np.ndarray):
                    v = key
                else:
                    try:
                        v = self[key]
                    except KeyError as ex:
                        if ignore_missing:
                            continue
                        raise ex
                matrix.append(v)
            matrix = np.array(matrix)
        if weights is not None:
            matrix = np.array(weights, copy=False).reshape(-1, 1) * matrix
        if pre_normalize:
            matrix = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix = np.mean(matrix, axis=0)
        if post_normalize:
            matrix = matrix / np.linalg.norm(matrix)
        return matrix

    def reset_index(self):
        self._index = None

    def rebuild_index(self, space="l2", ef_construction=200, M=16):
        num_elements, dim = self.shape
        p = hnswlib.Index(space=space, dim=dim)
        p.init_index(max_elements=num_elements, ef_construction=ef_construction, M=M)
        p.add_items(self.data)
        self._index = p

    @property
    def index(self) -> hnswlib.Index:
        if self._index is None:
            self.rebuild_index()
        return self._index

    def knn_query(self, data: np.ndarray, k=1):
        # ndarray is a 2d vector
        if len(data.shape) == 1:
            data = data.reshape(1, -1)
            labels, distances = self.index.knn_query(data, k=k)
            return (labels[0], distances[0])
        return self.index.knn_query(data, k=k)
