from __future__ import annotations
from collections.abc import Mapping
import typing

import pandas as pd
from torch.utils.data import IterableDataset
from datasets import Dataset as HuggingFaceDataset

from tklearn.nn.utils import to_tensor, get_index

__all__ = [
    "TrainerDataset",
]


class TrainerDataset(IterableDataset):
    """Dataset for PyTorch models."""

    def __init__(self, **kwargs):
        """Convert the data to a trainer compatible dataset.

        Parameters
        ----------
        kwargs : dict
            The data to use for the dataset.
        """
        datasets, length = {}, None
        for key, dataset in kwargs.items():
            if dataset is None:
                continue
            if isinstance(dataset, HuggingFaceDataset):
                # support for HuggingFace datasets
                num_rows = dataset.num_rows
            elif isinstance(dataset, pd.DataFrame):
                # support for pandas dataframes
                num_rows = dataset.shape[0]  # use len instead
            elif isinstance(dataset, Mapping):
                # support for mapping of datasets
                k = next(iter(dataset.keys()))
                num_rows = len(dataset[k])
            else:
                # support for other torch style datasets
                # e.g. list of dicts (mapping style)
                num_rows = len(dataset)
            if length is None:
                length = num_rows
            else:
                assert length == num_rows, (
                    "all datasets must have the same length, "
                    f"found {num_rows}, required {length}"
                )
            datasets[key] = dataset
        if not length:
            raise ValueError("no datasets specified")
        self._datasets = datasets
        self._length: int = length

    def __getitem__(self, idx) -> typing.Mapping[str, typing.Any]:
        """Get the data at the specified index."""
        record = get_index(self._datasets, idx)
        record["index"] = idx
        return to_tensor(record)

    def __len__(self) -> int:
        """Get the length of the dataset."""
        return self._length

    def __iter__(self) -> typing.Generator[typing.Any, None, None]:
        """Iterate over the dataset."""
        for idx in range(len(self)):
            yield self[idx]
