from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Callable, Optional, TypeVar, Union

import torch
from transformers import get_scheduler as _get_scheduler

from tklearn.nn.loss import LossDict

_tensor_or_tensors_type = Union[torch.Tensor, Iterable[torch.Tensor]]

ClipGradNormType = Union[
    int, float, bool, dict[str, Any], Callable[[_tensor_or_tensors_type], None]
]


ModelInput = TypeVar("BatchInput")
ModelOutput = TypeVar("BatchOutput")
LossLike = Union[torch.Tensor, Mapping[str, torch.Tensor], LossDict, None]
LossFunctionType = Callable[[ModelInput, ModelOutput], LossLike]
PostprocessorFunctionType = Callable[[ModelInput, ModelOutput], dict[str, Any]]


def get_scheduler(
    name: str,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    steps_per_epoch: int,
    # either num_warmup_steps or warmup_proportion
    num_warmup_steps: Union[int, str, None] = None,
    warmup_proportion: Optional[float] = None,
    **kwargs: Any,
) -> torch.optim.lr_scheduler._LRScheduler:
    if isinstance(num_warmup_steps, float) and num_warmup_steps < 1:
        # automatically convert to proportion
        warmup_proportion = num_warmup_steps
        num_warmup_steps = None
    if warmup_proportion is None:
        if isinstance(num_warmup_steps, str):
            # e.g., "1 epoch", "1 batch"
            number, unit = num_warmup_steps.split()
            if unit == "epoch":
                num_warmup_steps = int(number) * steps_per_epoch
            elif unit == "batch":
                num_warmup_steps = int(number)
            else:
                raise ValueError(f"invalid unit '{unit}' for num_warmup_steps")
        elif isinstance(num_warmup_steps, (int, float)):
            pass
        elif num_warmup_steps is None:
            pass
        else:
            raise ValueError("num_warmup_steps must be a float or None")
    elif num_warmup_steps is None:
        num_warmup_steps = int(epochs * steps_per_epoch * warmup_proportion)
    else:
        msg = "num_warmup_steps and warmup_proportion cannot be used together"
        raise ValueError(msg)
    num_training_steps = epochs * steps_per_epoch
    return _get_scheduler(
        name=name,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        # warmup_proportion=warmup_proportion,
        scheduler_specific_kwargs=kwargs,
    )
