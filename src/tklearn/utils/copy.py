import copy
from typing import Any, Mapping, TypeVar, Union

import torch

from tklearn.utils.array import move_to_device

__all__ = [
    "deepcopy",
]

T = TypeVar("T", bound=Any)


def _get_device(x: Any) -> torch.device:
    if hasattr(x, "device"):
        return x.device
    elif isinstance(x, Mapping):
        return _get_device(next(iter(x.values())))
    msg = f"unable determine device for object of type {x.__class__.__name__}"
    raise TypeError(msg)


def deepcopy(x: T, device: Union[str, torch.device, None] = None) -> T:
    """Deep copy a model."""
    try:
        current_device = _get_device(x)
    except TypeError:
        return copy.deepcopy(x)
    if device is None:
        device = current_device
    elif isinstance(device, str):
        device = torch.device(device)
    if device != current_device:
        # if x is a module it will be inplace
        #   then it should be moved back to the
        #   original device
        x = move_to_device(x, device)
        model_copy = copy.deepcopy(x)
        x = move_to_device(x, current_device)
    else:
        model_copy = copy.deepcopy(x)
    return model_copy
