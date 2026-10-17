"""Calibrate the probabilities of a classifier with temperature scaling
(Guo et al., 2017, "On Calibration of Modern Neural Networks").

A classifier fit closely to its training data is often overconfident:
its probabilities are higher than its accuracy. Temperature scaling
divides its logits by one number, the temperature, fit on held-out data
to minimize the negative log-likelihood. Dividing by a positive number
keeps the order of the logits, so the predictions stay the same; only
the probabilities change.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tklearn.nn.module import Module

if TYPE_CHECKING:
    from tklearn.nn.trainer import Trainer

__all__ = [
    "Activation",
    "CalibratedModule",
    "calibrate",
    "fit_temperature",
]

#: How logits become probabilities: ``"softmax"`` over the classes of a
#: multiclass classifier, ``"sigmoid"`` for each logit of a binary
#: (one logit per example) or multilabel classifier.
Activation = Literal["softmax", "sigmoid"]


def calibrate(
    trainer: Trainer,
    dataloader: DataLoader,
    *,
    activation: Activation,
) -> CalibratedModule:
    """Calibrate the model of `trainer` on the labelled `dataloader`.

    Runs `Trainer.predict` over `dataloader`, fits the temperature of its
    ``"logits"`` to its ``"y_true"`` with `fit_temperature`, and returns
    the trainer's model wrapped in a `CalibratedModule`. The model's
    `predict_step` must return both, so the batches must have labels.
    Use held-out data, not the training data, whose logits the model was
    fit to. `activation` is how the model's logits become probabilities:
    ``"softmax"`` for a multiclass classifier, or a binary one with two
    logits per example; ``"sigmoid"`` for a binary classifier with one
    logit per example, or a multilabel one.

    Examples
    --------
    >>> calibrated = calibrate(trainer, valid_loader, activation="softmax")
    >>> calibrated.temperature
    tensor(2.7100)
    >>> Trainer(calibrated, metrics=metrics).evaluate(test_loader)
    """
    outputs = trainer.predict(dataloader)
    if not isinstance(outputs, Mapping) or "logits" not in outputs:
        msg = 'calibrate needs predict_step to return the "logits"'
        raise ValueError(msg)
    if "y_true" not in outputs:
        msg = (
            'calibrate needs predict_step to return the "y_true" of '
            "labelled batches"
        )
        raise ValueError(msg)
    temperature = fit_temperature(
        outputs["logits"],
        outputs["y_true"],
        activation=activation,
    )
    return CalibratedModule(trainer.model, temperature, activation)


def fit_temperature(
    logits: torch.Tensor, y_true: torch.Tensor, *, activation: Activation
) -> float:
    """The temperature that minimizes the negative log-likelihood of
    `y_true` under the probabilities of ``logits / temperature``.

    Parameters
    ----------
    logits : Tensor
        ``(n, num_classes)`` for a multiclass classifier, ``(n,)`` for a
        binary one with one logit per example, or ``(n, num_labels)``
        for a multilabel one.
    y_true : Tensor
        Class indices ``(n,)`` for a multiclass classifier, 0 or 1 of the
        shape of `logits` otherwise.
    activation : {"softmax", "sigmoid"}
        ``"softmax"`` for a multiclass classifier, ``"sigmoid"`` for a
        binary or multilabel one.
    """
    _check_shapes(logits, y_true, activation)
    logits = logits.detach().float().cpu()
    y_true = y_true.detach().cpu()
    if activation == "sigmoid":
        y_true = y_true.float()
    # the log of the temperature, so that the temperature stays positive
    log_t = torch.zeros((), requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [log_t], max_iter=100, line_search_fn="strong_wolfe"
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = _nll(logits / log_t.exp(), y_true, activation).mean()
        loss.backward()
        return loss

    optimizer.step(closure)
    return log_t.exp().item()


class CalibratedModule(Module):
    """A classifier with its logits divided by a temperature.

    Its `predict_step` runs the model's, then replaces ``"logits"`` with
    the scaled logits, ``"y_score"`` with their probabilities (in the
    model's shape: the probability of the positive class when the model
    gives it alone for two classes), and ``"loss"``, on labelled
    batches, with the negative log-likelihood of each example under
    them. ``"y_pred"`` and the other outputs stay as they are.

    It is for `Trainer.evaluate` and `predict`; training it would train
    the wrapped model, with the temperature fixed.

    Parameters
    ----------
    model : Module
        The classifier; its `predict_step` returns ``"logits"``.
    temperature : float
        What the logits are divided by.
    activation : {"softmax", "sigmoid"}
        How the model's logits become probabilities.
    """

    def __init__(
        self, model: Module, temperature: float, activation: Activation
    ) -> None:
        super().__init__()
        if temperature <= 0:
            msg = f"the temperature must be positive, got {temperature}"
            raise ValueError(msg)
        _check_activation(activation)
        self.model = model
        self.activation: Activation = activation
        self.register_buffer("temperature", torch.tensor(float(temperature)))

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.model(*args, **kwargs)

    def predict_step(self, batch: Any) -> dict[str, Any]:
        outputs = self.model.predict_step(batch)
        if not isinstance(outputs, Mapping) or "logits" not in outputs:
            msg = (
                f"{type(self.model).__name__}.predict_step must return the "
                '"logits" to be calibrated'
            )
            raise ValueError(msg)
        outputs = dict(outputs)
        logits = outputs["logits"] / self.temperature
        outputs["logits"] = logits
        if self.activation == "softmax":
            scores = logits.softmax(-1)
        else:
            scores = logits.sigmoid()
        if "y_score" in outputs:
            outputs["y_score"] = _like(scores, outputs["y_score"])
        if outputs.get("y_true") is not None:
            y_true = outputs["y_true"]
            if self.activation == "sigmoid":
                y_true = y_true.to(logits.dtype)
            outputs["loss"] = _nll(logits, y_true, self.activation)
        return outputs


def _check_activation(activation: Any) -> None:
    if activation not in ("softmax", "sigmoid"):
        msg = f"activation must be softmax or sigmoid, got {activation!r}"
        raise ValueError(msg)


def _check_shapes(
    logits: torch.Tensor, y_true: torch.Tensor, activation: Activation
) -> None:
    """Raise unless `logits` and `y_true` have the shapes `activation`
    takes: class indices with a logit per class for ``"softmax"``, 0/1
    targets of the shape of the logits for ``"sigmoid"``."""
    _check_activation(activation)
    if activation == "softmax":
        if (
            logits.ndim == 2
            and y_true.ndim == 1
            and len(y_true) == len(logits)
        ):
            return
        expected = "logits (n, num_classes) and class indices (n,)"
    else:
        if logits.shape == y_true.shape and logits.ndim in (1, 2):
            return
        expected = (
            "logits and 0/1 targets of the same shape, (n,) or (n, num_labels)"
        )
    msg = (
        f"{activation} expects {expected}; got logits "
        f"{tuple(logits.shape)} and y_true {tuple(y_true.shape)}"
    )
    raise ValueError(msg)


def _nll(
    logits: torch.Tensor, y_true: torch.Tensor, activation: Activation
) -> torch.Tensor:
    """The negative log-likelihood of each example."""
    if activation == "softmax":
        return F.cross_entropy(logits, y_true, reduction="none")
    losses = F.binary_cross_entropy_with_logits(
        logits, y_true, reduction="none"
    )
    # one per example: the mean over the labels of a multilabel example
    return losses.mean(-1) if losses.ndim == 2 else losses


def _like(scores: torch.Tensor, y_score: Any) -> torch.Tensor:
    """`scores` in the shape of the model's `y_score`."""
    shape = tuple(torch.as_tensor(y_score).shape)
    if tuple(scores.shape) == shape:
        return scores
    if scores.ndim == 2 and scores.shape[1] == 2 and shape == scores.shape[:1]:
        return scores[:, 1]  # the positive class of a binary classifier
    msg = (
        f"cannot give calibrated probabilities {tuple(scores.shape)} the "
        f"shape of the model's y_score {shape}"
    )
    raise ValueError(msg)
