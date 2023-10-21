from collections.abc import Mapping
from typing import Union, List

import numpy as np
import pandas as pd
import torch

__all__ = [
    "concat",
    "tolist",
]


def concat(a, b):
    """Concatenate two objects together on primary axis.

    Parameters
    ----------
    a : object
        First object to concatenate.
    b : object
        Second object to concatenate.

    Returns
    -------
    object
        Concatenated object.
    """
    if b is None:
        return a
    if a is None:
        return b
    if isinstance(a, np.ndarray):
        return np.concatenate((a, b), axis=0)
    if isinstance(a, torch.Tensor):
        return torch.cat((a, b), dim=0)
    if isinstance(a, (pd.DataFrame, pd.Series)):
        return pd.concat((a, b), axis=0, ignore_index=True)
    if isinstance(a, tuple):
        return tuple(concat(s, o) for s, o in zip(a, b))
    if isinstance(a, Mapping):
        return {key: concat(value, b[key]) for key, value in a.items()}
    # any other sequence type
    return a + b


def tolist(x: Union[List, np.ndarray, torch.Tensor]):
    if isinstance(x, list):
        return x
    elif hasattr(x, "numpy"):  # Checks for TF tensors without needing the import
        x = x.numpy()
    return x.tolist()
