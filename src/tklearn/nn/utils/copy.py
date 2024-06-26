import copy
from typing import TYPE_CHECKING, Any, Mapping, Union

import torch

from tklearn.nn.utils.array import move_to_device

if TYPE_CHECKING:
    from tklearn import nn


def _get_device(model: Any) -> torch.device:
    if isinstance(model, nn.Module):
        return model.device
    elif isinstance(model, Mapping):
        return _get_device(next(iter(model.values())))
    msg = (
        "model must be either a nn.Module or a Mapping,"
        f" got {model.__class__.__name__}"
    )
    raise TypeError(msg)


def deepcopy(
    model: Union[nn.Module, Mapping[str, torch.Tensor]],
    device: Union[str, torch.device, None] = None,
) -> nn.Module:
    """Deep copy a model."""
    current_device = _get_device(model)
    if device is None:
        device = current_device
    elif isinstance(device, str):
        device = torch.device(device)
    if device != current_device:
        move_to_device(model, device)
        model_copy = copy.deepcopy(model)
        move_to_device(model, current_device)
    else:
        model_copy = copy.deepcopy(model)
    return model_copy
