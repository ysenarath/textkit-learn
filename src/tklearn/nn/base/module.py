from __future__ import annotations

import re
from typing import Any, Callable, Generic, TypeVar

import torch
import torch.nn as nn
from typing_extensions import Self

from tklearn.nn.loss import LossLike

__all__ = [
    "Module",
]

BatchT = TypeVar("BatchT")
OutputT = TypeVar("OutputT")


class Module(nn.Module, Generic[BatchT, OutputT]):
    """Base class for models driven by `Trainer`, `Evaluator` and `Predictor`.

    `BatchT` is the batch type produced by the dataloader and `OutputT` the output type
    of `predict_step`. Subclasses implement:

    - `predict_step` (required): run the model on a batch.
    - `compute_loss` (for training and loss reporting): loss from a batch
      and its output.
    - `compute_metric_inputs` (for metrics): the arrays metrics read, such as
      ``y_true``, ``y_pred`` and ``y_score``.

    Override `training_step` only when the training forward pass differs from
    ``compute_loss(batch, predict_step(batch))``.
    """

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def predict_step(self, batch: BatchT) -> OutputT:
        """Run the model on one batch."""
        raise NotImplementedError

    def compute_loss(self, batch: BatchT, output: OutputT) -> LossLike:
        """Compute the loss for one batch from its `predict_step` output."""
        raise NotImplementedError

    def training_step(self, batch: BatchT) -> LossLike:
        """Compute the training loss for one batch."""
        return self.compute_loss(batch, self.predict_step(batch))

    def compute_metric_inputs(
        self, batch: BatchT, output: OutputT
    ) -> dict[str, Any]:
        """Return the metric inputs for one batch, keyed by input name."""
        raise NotImplementedError

    def compile(
        self,
        fullgraph: bool = False,
        dynamic: bool | None = None,
        backend: str | Callable[..., Any] = "inductor",
        mode: str | None = None,
        options: dict[str, str | int | bool] | None = None,
        disable: bool = False,
    ) -> Self:
        return torch.compile(
            self,
            fullgraph=fullgraph,
            dynamic=dynamic,
            backend=backend,
            mode=mode,
            options=options,
            disable=disable,
        )

    def freeze_layers(
        self, layers: list[str] | None = None, prefix: str = ""
    ) -> int:
        """
        Freeze layers in the model that match the given patterns.

        Parameters
        ----------
        layers : list of str, optional
            A list of layer names or patterns to freeze. Supports wildcards (*)
            and dot notation for nested layers. If None, no layers will be frozen.
        prefix : str, default=""
            An optional prefix to apply to all layer patterns.

        Returns
        -------
        int
            The number of parameters frozen.

        Raises
        ------
        ValueError
            If an invalid regex pattern is provided.

        Examples
        --------
        >>> model.freeze_layers(['encoder.*', 'encoder.layer.[0-8].*'])
        >>> model.freeze_layers(['layer_[1-3]'], prefix='transformer')

        Notes
        -----
        This method uses regular expressions to match layer names. Dots in layer
        names are treated as literal dots, while asterisks are treated as wildcards.
        """
        if not layers:
            return 0  # no layers to freeze
        # escape dots and convert asterisks to regex wildcards
        layers = [p.replace(".", r"\.").replace("*", ".*") for p in layers]
        pattern_regex = "|".join(layers)
        if prefix:
            pattern_regex = rf"{prefix}\.({pattern_regex})"
        # compile regex pattern
        try:
            pattern = re.compile(f"^{pattern_regex}$")
        except re.error as e:
            raise ValueError(str(e))
        # freeze parameters that match the pattern
        frozen_params = 0
        for name, param in self.named_parameters():
            if not pattern.match(name):
                continue
            param.requires_grad = False
            frozen_params += param.numel()
        # return number of frozen parameters
        return frozen_params
