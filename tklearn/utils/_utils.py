from collections.abc import Mapping
from collections import UserDict
from typing import Union, List, Dict, Any
from copy import deepcopy

import numpy as np
import pandas as pd
import torch

__all__ = [
    "concat",
    "tolist",
    "merge",
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


def merge(
    x: Dict,
    y: Dict,
    inplace: bool = False,
    exists_strategy: str = "replace",
):
    out = x
    if not inplace:
        out = deepcopy(out)
    for key, value in y.items():
        if key in out:
            out_value = out[key]
            if isinstance(out_value, Mapping):
                # merge inplace - assumes value is dict
                merge(
                    out_value,
                    value,
                    inplace=True,
                    exists_strategy=exists_strategy,
                )
            else:
                if exists_strategy == "raise":
                    raise ValueError(f"value for {key} exists")
                elif exists_strategy not in {"keep", "ignore"}:
                    out[key] = value  # replace
        else:
            # set
            out[key] = value
    return out


def tolist(x: Union[List, np.ndarray, torch.Tensor]):
    if isinstance(x, list):
        return x
    elif hasattr(x, "numpy"):  # Checks for TF tensors without needing the import
        x = x.numpy()
    return x.tolist()


class AttrDict(UserDict):
    def __getitem__(self, key: Any) -> Any:
        if key in self:
            return self.get(key)
        super(AttrDict, self).__getitem__(key)
