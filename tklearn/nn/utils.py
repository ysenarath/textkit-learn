from __future__ import annotations
from collections.abc import Mapping, Sequence
import typing
from typing import Any

from datasets import Dataset as HuggingFaceDataset
from torch.utils.data import IterableDataset

import torch
import pandas as pd
import numpy as np

__all__ = [
    "to_tensor",
    "get_index",
    "move_to_device",
]


def to_tensor(data: typing.Any) -> typing.Any:
    if isinstance(data, Mapping):
        # convert mapping to numpy array (values only)
        return {key: to_tensor(value) for key, value in data.items()}
    elif isinstance(data, Sequence) and not isinstance(data, str):
        return torch.tensor(data)
    elif isinstance(data, np.generic):
        return torch.tensor(data.item())
    else:
        return torch.tensor(data)


def get_index(
    data: typing.Union[typing.Sequence, typing.Mapping, pd.DataFrame],
    idx: int,
) -> typing.Any:
    """Get the data at the specified index."""
    if isinstance(data, pd.Series):
        # get row at specified index
        return data.iloc[idx]
    if isinstance(data, pd.DataFrame):
        # get row at specified index
        return data.iloc[idx].to_dict()
    if isinstance(data, Mapping):
        # get index for each value in mapping
        out = {}
        for key, value in data.items():
            try:
                out[key] = get_index(value, idx)
            except IndexError as _:
                raise IndexError(f"unable to extract index from {str(key)}")
        return out
    return data[idx]


def move_to_device(data: Any, device, detach=False, numpy=False) -> Any:
    """Move data to device.

    This function is a recursive function that will
    move all tensors in a mapping or sequence to
    the specified device.

    Notes
    -----
    If the data cannot be cannot be moved to the
    specified device, then the data is returned
    as is.

    Parameters
    ----------
    data : torch.Tensor or Mapping or Sequence or typing.Any
        The data to move to device.
    device : torch.device or str
        The device to move the data to.
    detach : bool, optional
        Whether to detach the data from the computation graph.
        Defaults to False.
    numpy : bool, optional
        Whether to convert the data to a numpy array.

    Returns
    -------
    torch.Tensor or Mapping or Sequence or typing.Any
        The data moved to device.
    """
    if torch.is_tensor(data):
        if detach:
            data = data.detach()
        data = data.to(device)
        if numpy:
            data = data.numpy()
        return data
    elif isinstance(data, Mapping):
        return {key: move_to_device(value, device) for key, value in data.items()}
    elif isinstance(data, Sequence) and not isinstance(data, str):
        # try moving each element to device
        # assuming that they are already tensors
        return [move_to_device(value, device) for value in data]
    # if we cannot move the data to device
    # then just return the data as is
    return data


class TorchDataset(IterableDataset):
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
