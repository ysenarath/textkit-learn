from __future__ import annotations

import itertools
from collections.abc import Mapping
from typing import Any, Union

import torch

__all__ = [
    "Loss",
    "Module",
]

#: What `Module.training_step` returns: a scalar loss, or a mapping whose
#: ``"loss"`` entry is optimized and whose other scalar entries are logged.
Loss = Union[torch.Tensor, Mapping[str, Union[torch.Tensor, float]]]


class Module(torch.nn.Module):
    """A model that `Trainer` can fit, evaluate and run.

    Subclasses implement `predict_step`, and `training_step` when the
    training loss is not the ``"loss"`` that `predict_step` returns.

    - `fit` minimizes the loss from `training_step`.
    - `evaluate` averages the ``"loss"`` from `predict_step` over the
      examples and passes its other entries to the metrics, e.g.
      ``y_true``, ``y_pred`` and ``y_score``.
    - `predict` concatenates the `predict_step` outputs of every batch.

    Batches arrive on the model's device. Steps run in the model's
    precision, under autocast when the trainer uses mixed precision.

    Examples
    --------
    >>> class Classifier(Module):
    ...     def __init__(self, n_features, n_classes):
    ...         super().__init__()
    ...         self.linear = torch.nn.Linear(n_features, n_classes)
    ...
    ...     def forward(self, x):
    ...         return self.linear(x)
    ...
    ...     def predict_step(self, batch):
    ...         logits = self(batch["x"])
    ...         outputs = {
    ...             "y_pred": logits.argmax(-1),
    ...             "y_score": logits.softmax(-1),
    ...         }
    ...         if "labels" in batch:  # absent at inference time
    ...             outputs["y_true"] = batch["labels"]
    ...             outputs["loss"] = F.cross_entropy(logits, batch["labels"])
    ...         return outputs
    """

    @property
    def device(self) -> torch.device:
        """Device of the first parameter or buffer; CPU if there are none."""
        for tensor in itertools.chain(self.parameters(), self.buffers()):
            return tensor.device
        return torch.device("cpu")

    def training_step(self, batch: Any) -> Loss:
        """Compute the training loss for one batch.

        Returns a scalar loss, or a mapping with the loss under ``"loss"``
        and other scalars to log, e.g. the terms of a combined loss. By
        default it returns the ``"loss"`` from `predict_step`, averaged
        over the batch when there is one per example.
        """
        outputs = self.predict_step(batch)
        if isinstance(outputs, Mapping) and outputs.get("loss") is not None:
            loss = outputs["loss"]
            if isinstance(loss, torch.Tensor) and loss.ndim == 1:
                return loss.mean()
            return loss
        msg = (
            f"{type(self).__name__} must implement training_step, or return "
            "a mapping with a 'loss' from predict_step"
        )
        raise NotImplementedError(msg)

    def predict_step(self, batch: Any) -> Any:
        """Run the model on one batch.

        Called with a whole batch; the outputs hold the batch's examples
        along their first dimension (a tensor of one prediction per
        example, a list of one string per example, ...). To be evaluated,
        return a mapping of metric inputs, and the loss under ``"loss"``
        to report it: either the mean over the batch, or one loss per
        example, which `evaluate` averages exactly on several processes.
        `predict` accepts any tensor, array, list, tuple or mapping of them.
        """
        msg = f"{type(self).__name__} must implement predict_step"
        raise NotImplementedError(msg)
