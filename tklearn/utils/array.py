"""Utilities for working with arrays and tensors."""

from __future__ import annotations

from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Self,
    Tuple,
    TypeVar,
    Union,
)

import numpy as np
import pandas as pd
import torch
from numpy import typing as nt
from torch import Tensor

__all__ = [
    "move_to_device",
    "to_numpy",
    "to_torch",
]

T = TypeVar("T")

RT = Union[Mapping[Any, "RT"], Tuple["RT", ...], nt.NDArray[Any], torch.Tensor]


def to_numpy(obj: object) -> RT:
    """Convert an object to a numpy array.

    Parameters
    ----------
    obj : object
        The object to convert to a numpy array.

    Returns
    -------
    output : Any
        The object converted to a numpy array.
    """
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy()
    elif isinstance(obj, Dict):
        return {k: to_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, Tuple):
        return tuple(to_numpy(v) for v in obj)
    return np.array(obj)


def to_torch(obj: object) -> RT:
    """Convert an object to a torch tensor.

    Parameters
    ----------
    obj : object
        The object to convert to a torch tensor.

    Returns
    -------
    output : Any
        The object converted to a torch tensor.
    """
    if isinstance(obj, Dict):
        return {k: to_torch(v) for k, v in obj.items()}
    elif isinstance(obj, Tuple):
        return tuple(to_torch(v) for v in obj)
    if not isinstance(obj, np.ndarray):
        obj = np.array(obj)
    return torch.from_numpy(obj)


def detach(obj: T) -> T:
    """Detach a torch tensor from its computation graph.

    Parameters
    ----------
    obj : T
        The object to detach from its computation graph.

    Returns
    -------
    output : T
        The object detached from its computation graph.
    """
    if isinstance(obj, torch.Tensor):
        return obj.detach()
    elif isinstance(obj, Dict):
        return {k: detach(v) for k, v in obj.items()}
    elif isinstance(obj, List):
        return [detach(v) for v in obj]
    elif isinstance(obj, Tuple):
        return tuple(detach(v) for v in obj)
    return obj


class MovableToDeviceMixin:
    """Mixin class for moving an object to a device."""

    def to(
        self,
        device: Union[str, torch.device],
        detach: bool = False,
    ) -> Self:
        """Move the object to a device.

        Parameters
        ----------
        device : Union[str, torch.device]
            The device to move the object to.

        Returns
        -------
        output : MovableToDevice
            The object moved to the device.
        """
        return move_to_device(self, device, detach=detach)


def move_to_device(
    obj: T,
    device: Union[str, torch.device],
) -> T:
    """Move an object to a device.

    If the object is a torch tensor or module with the provided device, it is
    returned as is. If the object is a mapping, list, or tuple, the function is
    applied recursively to each element in the object.

    Parameters
    ----------
    obj : object
        The object to move to the device.
    device : Union[str, torch.device]
        The device to move the object to.

    Returns
    -------
    output : object
        The object moved to the device.
    """
    if isinstance(obj, torch.nn.Module):
        return obj.to(device)
    elif isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, MovableToDeviceMixin):
        return obj.to(device)
    elif isinstance(obj, Dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, List):
        return [move_to_device(v, device) for v in obj]
    elif isinstance(obj, Tuple):
        return tuple(move_to_device(v, device) for v in obj)
    return obj


def concat(
    objs: Union[List[RT], Dict],
    /,
    axis: int = 0,
) -> Union[RT, Dict]:
    """Concatenate a list of objects along an axis.

    Parameters
    ----------
    objs : List[RT]
        The list of objects to concatenate.
    axis : int, optional
        The axis to concatenate along, by default 0.

    Returns
    -------
    output : RT | Dict
        The concatenated objects.
    """
    objs = [o for o in objs if o is not None]
    if len(objs) == 1:
        return objs[0]
    elif isinstance(objs[0], Dict):
        return {k: concat([o[k] for o in objs], axis=axis) for k in objs[0]}
    elif isinstance(objs[0], Tuple):
        return tuple(
            concat([o[i] for o in objs], axis=axis)
            for i in range(len(objs[0]))
        )
    elif isinstance(objs[0], np.ndarray):
        return np.concatenate(objs, axis=axis)
    elif isinstance(objs[0], torch.Tensor):
        return torch.cat(objs, dim=axis)
    msg = f"cannot concatenate objects of type '{type(objs[0]).__name__}'"
    raise ValueError(msg)


def length_of_first_array_like_in_nested_dict(data: Dict) -> int:
    """Return the length of the first list in a nested dictionary.

    Parameters
    ----------
    data : Dict
        The nested dictionary to search for the first list.

    Returns
    -------
    length : int
        The length of the first list in the nested dictionary.
    """
    for v in data.values():
        if isinstance(v, Mapping):
            return length_of_first_array_like_in_nested_dict(v)
        elif isinstance(v, Tensor):
            return v.size(0)
        elif isinstance(v, (np.ndarray, pd.DataFrame)):
            return v.shape[0]
        if isinstance(v, (List, pd.Series)):
            return len(v)
    return 0
