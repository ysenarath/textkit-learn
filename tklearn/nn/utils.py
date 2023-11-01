from collections.abc import Mapping, Sequence
import typing
from typing import Any

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
