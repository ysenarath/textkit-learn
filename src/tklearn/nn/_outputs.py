"""What the trainer makes of the outputs of `Module` steps: the loss and
logs of `training_step`, the per-example losses and concatenated outputs
of `predict_step`."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

import numpy as np
import torch

from tklearn.nn.module import Loss

__all__ = [
    "BatchLogs",
    "concat",
    "example_losses",
    "split_loss",
]


def example_losses(
    outputs: Mapping[str, Any], batch: Any, device: torch.device
) -> torch.Tensor:
    """The loss of each example of a batch, from the ``loss`` that
    `predict_step` returned: one per example as it is, or a scalar (the
    mean over the batch) repeated for each example of the batch, counted
    from the per-example outputs or, without one, from the batch.
    """
    loss = torch.as_tensor(outputs["loss"]).detach().to(device)
    if loss.ndim == 1:
        return loss
    if loss.ndim > 1:
        msg = (
            "predict_step must return a scalar loss or one loss per "
            f"example, got one of shape {tuple(loss.shape)}"
        )
        raise ValueError(msg)
    num_examples = _num_examples({
        k: v for k, v in outputs.items() if k != "loss"
    })
    if num_examples is None:
        num_examples = _num_examples(batch)
    if num_examples is None:
        msg = (
            "cannot count the examples of a batch without a tensor in it or "
            "in the predict_step outputs; return one loss per example instead"
        )
        raise ValueError(msg)
    return loss.repeat(num_examples)


def _num_examples(data: Any) -> int | None:
    """Examples in a batch or in per-example outputs: the first dimension
    of the first tensor or array in `data`; None when there is none."""
    if isinstance(data, (torch.Tensor, np.ndarray)):
        return len(data) if data.ndim else None
    if isinstance(data, Mapping):
        data = list(data.values())
    if isinstance(data, (tuple, list)):
        for item in data:
            num_examples = _num_examples(item)
            if num_examples is not None:
                return num_examples
    return None


def split_loss(
    loss: Loss,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
    """The loss to minimize and the values to log.

    Tensors are logged as detached 0-d tensors: converting them to floats
    would wait for the device on every batch.
    """
    if isinstance(loss, Mapping):
        if loss.get("loss") is None:
            msg = "training_step returned a mapping without a 'loss'"
            raise ValueError(msg)
        terms = dict(loss)
    else:
        terms = {"loss": loss}
    total = terms["loss"]
    if not isinstance(total, torch.Tensor) or total.numel() != 1:
        msg = f"the loss must be a scalar tensor, got {total!r}"
        raise ValueError(msg)
    logs: dict[str, torch.Tensor | float] = {}
    for key, value in terms.items():
        if isinstance(value, torch.Tensor) and value.numel() == 1:
            logs[key] = value.detach().reshape(())
            continue
        try:
            logs[key] = float(value)
        except (TypeError, ValueError, RuntimeError):
            msg = f"training_step returned {key!r}, which is not a scalar"
            raise ValueError(msg) from None
    return total, logs


class BatchLogs(Mapping[str, float]):
    """The logs of a training batch, read-only, each value converted to a
    float on first access.

    Converting a tensor waits for the device, so a batch whose logs no
    callback reads does not.
    """

    def __init__(self, values: Mapping[str, torch.Tensor | float]) -> None:
        self._values = values
        self._floats: dict[str, float] = {}

    def __getitem__(self, key: str) -> float:
        if key not in self._floats:
            self._floats[key] = float(self._values[key])
        return self._floats[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __repr__(self) -> str:
        return repr(dict(self))


def concat(chunks: list[Any]) -> Any:
    """Concatenate per-batch outputs of the same structure."""
    first = chunks[0]
    if first is None:
        return None
    if isinstance(first, torch.Tensor):
        _validate_tensor_chunks(first, chunks)
        return torch.cat(chunks)
    if isinstance(first, np.ndarray):
        return np.concatenate(chunks)
    if isinstance(first, Mapping):
        _validate_mapping_chunks(first, chunks)
        return {k: concat([chunk[k] for chunk in chunks]) for k in first}
    if isinstance(first, tuple):
        _validate_tuple_chunks(first, chunks)
        values = [concat(list(items)) for items in zip(*chunks)]
        if hasattr(first, "_fields"):  # namedtuple
            return type(first)(*values)
        return tuple(values)
    if isinstance(first, list):
        return [item for chunk in chunks for item in chunk]
    msg = f"cannot concatenate outputs of type {type(first).__name__}"
    raise TypeError(msg)


def _validate_mapping_chunks(first: Mapping, chunks: list[Mapping]) -> None:
    for chunk in chunks[1:]:
        missing = [k for k in first if k not in chunk]
        extra = [k for k in chunk if k not in first]
        if missing or extra:
            diffs = [f"lacks {missing}"] if missing else []
            diffs += [f"has the extra {extra}"] if extra else []
            msg = (
                "predict_step must return the same keys for every "
                f"batch; a batch {' and '.join(diffs)}"
            )
            raise ValueError(msg)


def _validate_tuple_chunks(first: tuple, chunks: list[tuple]) -> None:
    lengths = {len(chunk) for chunk in chunks}
    if len(lengths) > 1:
        msg = (
            "predict_step must return tuples of the same length for "
            f"every batch; got lengths {sorted(lengths)}"
        )
        raise ValueError(msg)


def _validate_tensor_chunks(
    first: torch.Tensor, chunks: list[torch.Tensor]
) -> None:
    if first.ndim == 0:
        msg = (
            "cannot concatenate scalar outputs; predict_step must return "
            "one value per example"
        )
        raise ValueError(msg)
