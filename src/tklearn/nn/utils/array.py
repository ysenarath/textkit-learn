"""Utilities for working with arrays and tensors."""

from __future__ import annotations

import functools
import operator
from typing import (
    Any,
    Dict,
    Generator,
    List,
    Mapping,
    NamedTuple,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    overload,
)

import numpy as np
import pandas as pd
import torch
from datasets import Dataset as HuggingFaceDataset
from numpy import typing as nt
from typing_extensions import Self

__all__ = [
    "move_to_device",
    "to_numpy",
    "to_torch",
]

try:
    from octoflow.data import Dataset as OctoFlowDataset
except ImportError:
    OctoFlowDataset = list

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
    elif isinstance(obj, Tuple) and hasattr(obj, "_fields"):
        # namedtuple
        dtype = type(obj)
        return dtype(to_numpy(v) for v in obj)
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
    elif isinstance(obj, Tuple) and hasattr(obj, "_fields"):
        # namedtuple
        dtype = type(obj)
        return dtype(to_torch(v) for v in obj)
    elif isinstance(obj, Tuple) and hasattr(obj, "_fields"):
        # namedtuple
        dtype = type(obj)
        return dtype(to_torch(v) for v in obj)
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
    elif isinstance(obj, Tuple) and hasattr(obj, "_fields"):
        # namedtuple
        dtype = type(obj)
        return dtype(detach(v) for v in obj)
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
    elif isinstance(obj, Mapping):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, Tuple) and hasattr(obj, "_fields"):
        # namedtuple
        dtype = type(obj)
        return dtype(move_to_device(v, device) for v in obj)
    elif isinstance(obj, Tuple):
        return tuple(move_to_device(v, device) for v in obj)
    elif isinstance(obj, List):
        return [move_to_device(v, device) for v in obj]
    return obj


def concat(objs: List[RT], /, axis: int = 0) -> RT:
    """Concatenate a list of objects along an axis.

    Parameters
    ----------
    objs : List[RT]
        The list of objects to concatenate.
    axis : int, optional
        The axis to concatenate along, by default 0.

    Returns
    -------
    output : RT
        The concatenated objects.
    """
    if len(objs) == 0:
        msg = "cannot concatenate empty list of objects"
        raise ValueError(msg)
    # remove None objects from list
    objs = [o for o in objs if o is not None]
    if len(objs) == 0:
        # if all objects were None
        return None
    elem = objs[0]
    if len(objs) == 1:
        return elem
    if isinstance(elem, Mapping):
        keys = set(elem.keys())
        output = {k: concat([o[k] for o in objs], axis=axis) for k in keys}
        try:
            # try to convert to original type assuming dict style constructor
            return type(elem)(**output)
        except TypeError:
            # if not possible, return as a dictionary
            return output
    if isinstance(elem, Tuple) and hasattr(elem, "_fields"):  # namedtuple
        dtype = type(elem)
        size = len(elem)
        # difference with normal tuple is that we return the same type as the input
        return dtype(concat([o[i] for o in objs], axis=axis) for i in range(size))
    if isinstance(elem, Tuple):
        size = len(elem)
        return tuple(concat([o[i] for o in objs], axis=axis) for i in range(size))
    if isinstance(elem, np.ndarray):
        return np.concatenate(objs, axis=axis)
    if isinstance(elem, torch.Tensor):
        return torch.cat(objs, dim=axis)
    if isinstance(elem, List):
        return functools.reduce(operator.iadd, objs, [])
    msg = f"cannot concatenate objects of type '{type(elem).__name__}'"
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
    if isinstance(data, (HuggingFaceDataset, OctoFlowDataset)):
        return len(data)
    if isinstance(data, (np.ndarray, pd.DataFrame)):
        return data.shape[0]
    if isinstance(data, torch.Tensor):
        return data.size(0)
    if isinstance(data, Tuple):
        # namedtuple check => hasattr(data, "_fields")
        # tuple or namedtuple
        for v in data:
            return length_of_first_array_like_in_nested_dict(v)
        return 0
    if isinstance(data, (Sequence, pd.Series)) and not isinstance(data, str):
        # any other sequence
        return len(data)
    if isinstance(data, Mapping):
        for v in data.values():
            return length_of_first_array_like_in_nested_dict(v)
        return 0
    # otherwise, it should be a dict
    msg = f"expected data type array-like or dict, got {type(data).__name__}"
    raise TypeError(msg)


D = TypeVar("D", Dict, Tuple, NamedTuple)

Indexable = Union[
    List,
    pd.DataFrame,
    pd.Series,
    np.ndarray,
    torch.Tensor,
    HuggingFaceDataset,
    OctoFlowDataset,
    Dict,
    Tuple,
]


@overload
def get_index(data: List[T], index: int) -> T: ...
@overload
def get_index(data: List[T], index: slice) -> List[T]: ...
@overload
def get_index(data: pd.DataFrame, index: int) -> pd.Series: ...
@overload
def get_index(data: pd.DataFrame, index: slice) -> pd.DataFrame: ...
@overload
def get_index(data: pd.Series, index: int) -> Any: ...
@overload
def get_index(data: pd.Series, index: slice) -> pd.Series: ...


def get_index(data: Any, index: Union[int, slice]) -> Any:
    """Get the value at a specific index in a nested dictionary.

    Parameters
    ----------
    data : Any
        The nested dictionary to search for the value at the index.
    index : int | slice
        The index to search for in the nested dictionary.

    Returns
    -------
    value : Any
        The value at the index in the nested dictionary.
    """
    if isinstance(
        data, (torch.Tensor, np.ndarray, HuggingFaceDataset, OctoFlowDataset)
    ):
        # will return dtyped value
        return data[index]
    if isinstance(data, (pd.DataFrame, pd.Series)):
        # will return dtyped value
        return data.iloc[index]
    if isinstance(data, Mapping):
        return {k: get_index(v, index=index) for k, v in data.items()}
    if isinstance(data, Tuple) and hasattr(data, "_fields"):
        # namedtuple
        dtype = type(data)
        return dtype(*(get_index(item, index=index) for item in data))
    if isinstance(data, Tuple):
        return tuple(get_index(item, index=index) for item in data)
    if isinstance(data, Sequence) and not isinstance(data, str):
        # will return dtyped value
        return data[index]
    msg = f"cannot get index from object of type '{type(data).__name__}'"
    raise ValueError(msg)


def batched(data: T, batch_size: int) -> Generator[T, None, None]:
    """Yield batches of data from a list or dictionary.

    Parameters
    ----------
    data : Any
        The data to yield batches from.
    batch_size : int
        The size of each batch.

    Yields
    ------
    batch : Any
        A batch of data from the input data.
    """
    for i in range(0, len(data), batch_size):
        yield get_index(data, slice(i, i + batch_size))
