from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable, TypeVar, Union

import torch
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from tklearn.nn.utils.collections import TensorDict
from tklearn.utils.targets import TargetType, type_of_target

__all__ = [
    "LossDict",
    "LossFunction",
    "LossLike",
    "TargetBasedLoss",
]

BatchT = TypeVar("BatchT")
OutputT = TypeVar("OutputT")


class LossDict(TensorDict):
    """Named loss terms that behave like a single loss.

    A bare tensor is stored under the key ``"loss"``. `backward` sums all
    terms, so a model can return several losses and have each one logged.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        if len(args) == 1 and not isinstance(args[0], Mapping):
            args = ({"loss": args[0]},)
        super().__init__(*args, **kwargs)


#: A scalar loss tensor, or named loss terms.
LossLike = Union[torch.Tensor, Mapping[str, torch.Tensor], LossDict]

#: ``loss(batch, output)`` -> loss, used to override `Module.compute_loss`.
LossFunction = Callable[[BatchT, OutputT], LossLike]


class TargetBasedLoss(torch.nn.Module):
    """The standard loss for a target type, applied to logits.

    - ``continuous``, ``continuous-multioutput``: `MSELoss`
    - ``multiclass``: `CrossEntropyLoss`
    - ``binary``, ``multilabel-indicator``: `BCEWithLogitsLoss`

    The target is reshaped and cast to match the input: multiclass targets
    are class indices with one fewer dimension than the input, and all other
    targets have as many elements as the input.

    Parameters
    ----------
    target_type : str or TargetType
        One of the target types in the table above.
    **kwargs
        Passed to the loss, e.g. ``weight``, ``pos_weight``,
        ``label_smoothing`` or ``ignore_index``.
    """

    def __init__(self, target_type: str | TargetType, **kwargs: Any):
        super().__init__()
        self.target_type = type_of_target(target_type)
        label = self.target_type.label
        if label in ("continuous", "continuous-multioutput"):
            self._loss_func = MSELoss(**kwargs)
        elif label == "multiclass":
            self._loss_func = CrossEntropyLoss(**kwargs)
        elif label in ("binary", "multilabel-indicator"):
            self._loss_func = BCEWithLogitsLoss(**kwargs)
        else:
            msg = f"target type '{label}' is not supported"
            raise ValueError(msg)

    def forward(
        self, input: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        if isinstance(self._loss_func, CrossEntropyLoss):
            return self._loss_func(
                input.reshape(-1, input.size(-1)), target.reshape(-1).long()
            )
        target = target.reshape(input.shape).to(dtype=input.dtype)
        return self._loss_func(input, target)
