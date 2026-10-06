from __future__ import annotations

import math
from typing import Any

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

__all__ = [
    "get_scheduler",
]


def get_scheduler(
    name: str,
    optimizer: Optimizer,
    num_training_steps: int,
    *,
    warmup: int | float = 0,
    **kwargs: Any,
) -> LRScheduler:
    """Build a named learning rate schedule with warmup.

    Parameters
    ----------
    name : str
        A `transformers` schedule: ``"linear"``, ``"cosine"``,
        ``"cosine_with_restarts"``, ``"polynomial"``, ``"constant"``,
        ``"constant_with_warmup"``, ``"inverse_sqrt"``, ...
    optimizer : Optimizer
        The optimizer whose learning rate is scheduled.
    num_training_steps : int
        Total number of optimizer steps.
    warmup : int or float, default=0
        Warmup length: a number of steps (int), or a fraction of
        `num_training_steps` (float in ``[0, 1]``).
    **kwargs
        Schedule-specific arguments, e.g. ``num_cycles`` or ``power``.

    Returns
    -------
    LRScheduler
        A scheduler to step after every optimizer step.
    """
    from transformers import get_scheduler as get_hf_scheduler

    return get_hf_scheduler(
        name,
        optimizer,
        num_warmup_steps=warmup_steps(warmup, num_training_steps),
        num_training_steps=num_training_steps,
        scheduler_specific_kwargs=kwargs,
    )


def warmup_steps(warmup: int | float, num_training_steps: int) -> int:
    """Number of warmup steps for a step count or a fraction."""
    if isinstance(warmup, bool) or not isinstance(warmup, (int, float)):
        msg = f"warmup must be an int or a float, got {warmup!r}"
        raise TypeError(msg)
    if isinstance(warmup, float):
        if not 0 <= warmup <= 1:
            msg = f"a float warmup must be in [0, 1], got {warmup}"
            raise ValueError(msg)
        return math.ceil(warmup * num_training_steps)
    if warmup < 0:
        msg = f"warmup must be non-negative, got {warmup}"
        raise ValueError(msg)
    return warmup
