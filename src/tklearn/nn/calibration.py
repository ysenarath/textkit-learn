"""Calibrate the probabilities of a classifier with temperature scaling
(Guo et al., 2017, "On Calibration of Modern Neural Networks").

A classifier fit closely to its training data is often overconfident:
its probabilities are higher than its accuracy. Temperature scaling
divides its logits by one number, the temperature, fit on held-out data
to minimize the negative log-likelihood. Dividing by a positive number
keeps the order of the logits, so the predictions stay the same; only
the probabilities change. A multilabel classifier can have one
temperature per label instead, as each label is a binary problem of
its own.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
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
    per_label: bool = False,
    shrinkage: float = 1.0,
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
    logit per example, or a multilabel one. `per_label` and `shrinkage`
    fit one temperature per label of a multilabel classifier (see
    `fit_temperature`).

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
        per_label=per_label,
        shrinkage=shrinkage,
    )
    return CalibratedModule(trainer.model, temperature, activation)


def fit_temperature(
    logits: torch.Tensor,
    y_true: torch.Tensor,
    *,
    activation: Activation,
    per_label: bool = False,
    shrinkage: float = 1.0,
) -> torch.Tensor:
    """The temperature that minimizes the negative log-likelihood of
    `y_true` under the probabilities of ``logits / temperature``: a
    scalar, or one per label ``(num_labels,)`` with `per_label`.

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
    per_label : bool, default False
        Fit one temperature per label of a multilabel classifier, with
        ``"sigmoid"`` and logits ``(n, num_labels)``. A label's
        predictions stay the same at the 0.5 threshold.
    shrinkage : float, default 1.0
        With `per_label`, how strongly each label's temperature is
        pulled toward the one temperature shared by all labels, which is
        fit first. Without it, the temperature of a label whose examples
        do not settle it (no positives, or a few positives all above the
        negatives) tends to 0, and its probabilities to 0 or 1. The pull
        weakens as the count of a label's rarer class, positives or
        negatives, grows: by default it holds a label with none or a few
        near the shared temperature, and barely moves one with hundreds.
        0 fits each label alone.
    """
    _check_shapes(logits, y_true, activation)
    if per_label and (activation != "sigmoid" or logits.ndim != 2):
        msg = (
            "per_label needs the logits (n, num_labels) of a multilabel "
            'classifier, with activation="sigmoid"'
        )
        raise ValueError(msg)
    if shrinkage < 0:
        msg = f"shrinkage must be at least 0, got {shrinkage}"
        raise ValueError(msg)
    logits = logits.detach().float().cpu()
    y_true = y_true.detach().cpu()
    if activation == "sigmoid":
        y_true = y_true.float()
    # fit the log of the temperature, so that the temperature stays positive
    shared = _minimize(
        torch.zeros(()),
        lambda log_t: _nll(logits / log_t.exp(), y_true, activation).mean(),
    )
    if not per_label:
        return shared.exp()
    # what settles a label's temperature is how many examples its rarer
    # class has, not how many it has in all: a label with 3 positives
    # among thousands of easy negatives is barely settled
    minority = torch.minimum(y_true.sum(0), (1 - y_true).sum(0))
    weight = shrinkage / (1 + minority)

    def loss(log_t: torch.Tensor) -> torch.Tensor:
        # each label's mean negative log-likelihood, plus its pull toward
        # the shared temperature; the labels' terms are independent
        nll = F.binary_cross_entropy_with_logits(
            logits / log_t.exp(), y_true, reduction="none"
        ).mean(0)
        return (nll + weight * (log_t - shared) ** 2).sum()

    return _minimize(shared.repeat(logits.shape[1]), loss).exp()


def _minimize(
    log_t: torch.Tensor, loss: Callable[[torch.Tensor], torch.Tensor]
) -> torch.Tensor:
    """The `log_t` that minimizes `loss`, starting from `log_t`."""
    log_t = log_t.clone().requires_grad_()
    optimizer = torch.optim.LBFGS(
        [log_t], max_iter=100, line_search_fn="strong_wolfe"
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        value = loss(log_t)
        value.backward()
        return value

    optimizer.step(closure)
    return log_t.detach()


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
    temperature : float or Tensor
        What the logits are divided by: a scalar, or one per label
        ``(num_labels,)`` of a multilabel classifier with ``"sigmoid"``.
    activation : {"softmax", "sigmoid"}
        How the model's logits become probabilities.
    """

    def __init__(
        self,
        model: Module,
        temperature: float | torch.Tensor,
        activation: Activation,
    ) -> None:
        super().__init__()
        _check_activation(activation)
        temperature = torch.as_tensor(temperature).detach().float().cpu()
        if temperature.ndim > 1 or (
            temperature.ndim == 1 and activation != "sigmoid"
        ):
            msg = (
                "the temperature must be a scalar, or one per label with "
                f'activation="sigmoid"; got {tuple(temperature.shape)} with '
                f"{activation!r}"
            )
            raise ValueError(msg)
        if (temperature <= 0).any():
            msg = f"the temperature must be positive, got {temperature}"
            raise ValueError(msg)
        self.model = model
        self.activation: Activation = activation
        self.register_buffer("temperature", temperature.clone())

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
        logits = outputs["logits"]
        if self.temperature.ndim == 1 and (
            logits.ndim != 2 or logits.shape[1] != len(self.temperature)
        ):
            msg = (
                f"{len(self.temperature)} temperatures, one per label, "
                f"cannot scale logits {tuple(logits.shape)}"
            )
            raise ValueError(msg)
        logits = logits / self.temperature
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
