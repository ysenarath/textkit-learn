from collections.abc import Mapping, Sequence
import typing

import torch
import pandas as pd
import numpy as np

__all__ = [
    "to_tensor",
    "get_index",
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
    data: typing.Union[typing.Sequence, typing.Mapping], idx: int
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
        return {key: get_index(value, idx) for key, value in data.items()}
    return data[idx]
