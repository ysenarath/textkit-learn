from __future__ import annotations

import json
import math
import os
from collections.abc import KeysView, Mapping
from pathlib import Path
from typing import Any, Optional, overload

import faiss
import numpy as np
from rich.progress import track


class TextEmbeddingIndex(Mapping[str, np.ndarray]):
    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.exists():
            self.path.mkdir(parents=True)
        self.index_path = self.path / "index.faiss"
        self.str2index_path = self.path / "str2index.json"
        self.index2str_path = self.path / "index2str.json"
        self.vals_path = self.path / "values.npy"
        self.str2index = {}
        self.vectors = []
        self.load()

    def load(self):
        if os.path.exists(self.str2index_path) and os.path.exists(
            self.vals_path
        ):
            with open(self.str2index_path, "r") as file:
                self.str2index = json.load(file)
            arr: np.ndarray = np.load(self.vals_path)
            self.vectors = [arr[i] for i in range(arr.shape[0])]
        self._index = None
        self._index2str = None
        self._updated = False

    def __setitem__(self, key: str, value: np.ndarray):
        if key not in self.str2index:
            index = 0
            if self.vectors is not None:
                index = len(self.vectors)
            self.str2index[key] = index
            self.vectors += [value]
        else:
            self.vectors[self.str2index[key]] = value
        self._updated = True

    def __getitem__(self, key: str | int) -> np.ndarray:
        if isinstance(key, str):
            return self.vectors[self.str2index[key]]
        return self.vectors[key]

    def __contains__(self, key: str) -> bool:
        return key in self.str2index

    def __len__(self) -> int:
        return len(self.str2index)

    def keys(self) -> KeysView:
        return self.str2index.keys()

    def values(self) -> list[np.ndarray]:
        return self.vectors

    def save(self):
        if self._updated:
            self.reset()
        with open(self.str2index_path, "w") as file:
            json.dump(self.str2index, file)
        np.save(self.vals_path, self.vectors)

    def _create_index(self):
        vectors = np.array(self.vectors)
        # Normalize the vectors
        faiss.normalize_L2(vectors)
        # Get the dimension of the vectors
        N, D = vectors.shape
        # Setup the index
        hnsw_m = 32  # The number of neighbors for HNSW. This is typically 32
        quantizer = faiss.IndexHNSWFlat(D, hnsw_m, faiss.METRIC_INNER_PRODUCT)
        # nlist - The number of cells (space partition). Typical value is sqrt(N)
        nlist = math.floor(max(math.sqrt(N), 1))
        M = 16  # The number of sub-vector. Typically this is 8, 16, 32, etc.
        nbits = 8  # bits per sub-vector. This is typically 8, so that each sub-vec is encoded by 1 byte
        index = faiss.IndexIVFPQ(quantizer, D, nlist, M, nbits)
        # Train
        K = min(256 * nlist, vectors.shape[0])
        Xt = vectors[:K]  # K vectors for training
        index.train(Xt)
        desc = "Adding vectors to index"
        batch_size = 256
        batches = np.arange(0, vectors.shape[0], batch_size)
        for batch_start in track(batches, description=desc):
            batch = vectors[batch_start : batch_start + batch_size]
            batch_ids = np.arange(
                batch_start, batch_start + len(batch), dtype=np.int64
            )
            index.add_with_ids(batch, batch_ids)
        faiss.write_index(index, str(self.index_path))

    def _setup_index(self):
        index_path = str(self.index_path)
        if not os.path.exists(index_path):
            self._create_index()
        self._index = faiss.read_index(index_path)

    @property
    def index(self) -> faiss.Index:
        if self._updated:
            self.reset()
        # freeze the index2str
        _ = self.index2str
        if getattr(self, "_index", None) is None:
            self._setup_index()
        return self._index

    def _setup_index2str(self):
        # Setup the index2str
        if os.path.exists(self.index2str_path):
            index2str = {}
            with open(self.index2str_path, "r") as file:
                for k, v in json.load(file).items():
                    index2str[int(k)] = v
        else:
            # Create the index2str
            index2str = {int(v): k for k, v in self.str2index.items()}
            # Write the index2str to a file
            with open(self.index2str_path, "w") as file:
                json.dump(index2str, file)
        self._index2str = index2str

    @property
    def index2str(self) -> dict:
        if self._updated:
            self.reset()
        if getattr(self, "_index2str", None) is None:
            self._setup_index2str()
        return self._index2str

    def reset(self):
        # delete the index and index2str
        if os.path.exists(self.index_path):
            os.remove(self.index_path)
        if os.path.exists(self.index2str_path):
            os.remove(self.index2str_path)
        self._index = None
        self._index2str = None
        self._updated = False

    def get_index(self, string: Optional[str]) -> Optional[int]:
        if string is None or len(string) == 0:
            return None
        return self.str2index[string]

    @overload
    def search(
        self, vectors: np.ndarray, k: int = 5
    ) -> list[tuple[str, int]]: ...
    @overload
    def search(
        self, vectors: np.ndarray, k: int = 5, return_index: bool = True
    ) -> list[tuple[int, int]]: ...
    @overload
    def search(
        self, vectors: list[np.ndarray], k: int = 5
    ) -> list[list[tuple[str, int]]]: ...
    @overload
    def search(
        self, vectors: list[np.ndarray], k: int = 5, return_index: bool = True
    ) -> list[list[tuple[int, int]]]: ...
    def search(
        self, vectors: Any, k: int = 5, return_index: bool = False
    ) -> Any:
        is_single = isinstance(vectors, np.ndarray) and vectors.ndim == 1
        if is_single:
            vectors = [vectors]
        # Encode the queries
        query_vectors = vectors
        # Normalize the vectors
        faiss.normalize_L2(query_vectors)
        # Search the index
        distances, indices = self.index.search(query_vectors, k)
        results = []
        for i in range(len(query_vectors)):
            result = []
            for j, index in enumerate(indices[i]):
                dist = distances[i][j].item()
                key = index if return_index else self.index2str[index]
                result.append((key, dist))
            results.append(result)
        if is_single:
            return results[0]
        return results
