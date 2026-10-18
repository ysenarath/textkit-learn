"""Text classifiers over Hugging Face Transformers models, ready for
`Trainer`, `tklearn.metrics` and `tklearn.nn.calibration`.

`SequenceClassifier` classifies a text (or a pair) into classes, labels
or values; `TokenClassifier` tags each token, e.g. for named entities.
Both take tokenized batches, e.g. from a `DataCollatorWithPadding` or a
`DataCollatorForTokenClassification`, with the model's inputs
(``input_ids``, ``attention_mask``, ...) and, to fit and evaluate,
``labels``; other entries of a batch are ignored. They compute their
loss themselves, so the model's own loss is never used.
"""

from __future__ import annotations

import inspect
from collections.abc import Sequence
from typing import Any, Literal

import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    PreTrainedModel,
)

from tklearn.nn._outputs import PAD_INDEX
from tklearn.nn.calibration import Activation
from tklearn.nn.module import Module

__all__ = [
    "SequenceClassifier",
    "Task",
    "TokenClassifier",
]

#: What a `SequenceClassifier` predicts: one class of several
#: (``"multiclass"``), one of two from a single logit (``"binary"``), any
#: number of labels (``"multilabel"``), or values (``"regression"``).
Task = Literal["multiclass", "binary", "multilabel", "regression"]

_TASKS = ("multiclass", "binary", "multilabel", "regression")


class SequenceClassifier(Module):
    """A Hugging Face sequence classification model, for `Trainer`.

    Its `predict_step` returns the ``logits``, ``y_pred`` and, except for
    regression, ``y_score``, and on labelled batches ``y_true`` and the
    ``loss`` of each example:

    - ``multiclass``: ``(n, k)`` logits, their argmax and softmax, and
      the cross-entropy.
    - ``binary``: ``(n,)`` logits, 1 where positive and their sigmoid,
      and the binary cross-entropy.
    - ``multilabel``: ``(n, k)`` logits, 1 where positive and their
      sigmoid, and the mean binary cross-entropy over the labels.
    - ``regression``: ``(n,)`` logits, or ``(n, k)`` for several
      outputs, as ``y_pred``, and the mean squared error over them.

    ``labels`` are class indices for multiclass, 0 or 1 for binary, an
    indicator matrix for multilabel and values for regression, as the
    metrics of `tklearn.metrics` read them.

    Parameters
    ----------
    model : str or PreTrainedModel
        A model name or path, loaded with
        `AutoModelForSequenceClassification.from_pretrained` with as many
        outputs as the task needs, or a loaded model with them.
    num_labels : int, optional
        Classes for multiclass, labels for multilabel, and outputs for
        regression (1 by default). Binary classification has one logit,
        and takes none. Read from a loaded model when not given.
    task : {"multiclass", "binary", "multilabel", "regression"}, \
default="multiclass"
        What the model predicts.
    **kwargs
        Passed to `from_pretrained`, e.g. ``id2label`` or ``revision``.

    Examples
    --------
    >>> model = SequenceClassifier("bert-base-uncased", 3)
    >>> trainer = Trainer(model, optimizer, metrics={"f1": F1("macro")})
    >>> trainer.fit(train_loader, valid_loader, epochs=3)
    >>> calibrate(trainer, valid_loader, activation=model.activation)
    """

    def __init__(
        self,
        model: str | PreTrainedModel,
        num_labels: int | None = None,
        *,
        task: Task = "multiclass",
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if task not in _TASKS:
            msg = f"task must be one of {', '.join(_TASKS)}, got {task!r}"
            raise ValueError(msg)
        if task == "binary" and num_labels is not None:
            msg = "a binary classifier has one logit, and takes no num_labels"
            raise ValueError(msg)
        if task == "binary":
            num_labels = 1
        elif (
            task == "regression"
            and num_labels is None
            and isinstance(model, str)
        ):
            num_labels = 1
        self.model = _load(
            AutoModelForSequenceClassification, model, num_labels, kwargs
        )
        self.task: Task = task
        self.num_labels: int = self.model.config.num_labels
        if task == "multiclass" and self.num_labels < 2:
            msg = (
                f"a multiclass classifier needs 2 or more classes, got "
                f'{self.num_labels}; use task="binary" for one logit'
            )
            raise ValueError(msg)
        self._inputs = _input_names(self.model)

    @property
    def activation(self) -> Activation | None:
        """How the logits become probabilities, for `calibrate`; None for
        regression."""
        if self.task == "multiclass":
            return "softmax"
        if self.task == "regression":
            return None
        return "sigmoid"

    def forward(self, **inputs: Any) -> torch.Tensor:
        """The logits of the model's inputs, as in the table above."""
        logits = self.model(**inputs).logits
        if self.num_labels == 1 and self.task in ("binary", "regression"):
            logits = logits.squeeze(-1)
        return logits

    def predict_step(self, batch: Any) -> dict[str, Any]:
        logits = self(**_inputs(batch, self._inputs))
        outputs: dict[str, Any] = {"logits": logits}
        if self.task == "multiclass":
            outputs["y_pred"] = logits.argmax(-1)
            outputs["y_score"] = logits.softmax(-1)
        elif self.task == "regression":
            outputs["y_pred"] = logits
        else:
            outputs["y_pred"] = (logits > 0).long()
            outputs["y_score"] = logits.sigmoid()
        labels = batch.get("labels") if hasattr(batch, "get") else None
        if labels is not None:
            outputs["y_true"] = labels
            outputs["loss"] = self._losses(logits, labels)
        return outputs

    def _losses(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """The loss of each example."""
        if self.task == "multiclass":
            return F.cross_entropy(logits, labels, reduction="none")
        if self.task == "regression":
            losses = F.mse_loss(
                logits, labels.to(logits.dtype), reduction="none"
            )
        else:
            losses = F.binary_cross_entropy_with_logits(
                logits, labels.to(logits.dtype), reduction="none"
            )
        # one per example: the mean over the labels or outputs
        return losses.mean(-1) if losses.ndim == 2 else losses


class TokenClassifier(Module):
    """A Hugging Face token classification model, for `Trainer`.

    ``labels`` hold the class of each token, and -100 on the tokens that
    do not count, e.g. padding, special tokens and the subwords after a
    word's first, as `DataCollatorForTokenClassification` pads them. The
    ``loss`` is the mean cross-entropy over the tokens that count, as
    Transformers computes it.

    Its `predict_step` returns the ``logits`` ``(n, length, k)``, a bool
    ``mask`` of the tokens that count (those with a label, or without
    labels, those of the attention mask) and ``y_pred``, and on labelled
    batches ``y_true`` and the ``loss``. With `label_names` (e.g. BIO
    tags), ``y_pred`` and ``y_true`` are lists of the
    tags of each sequence's tokens that count, as `SpanF1` reads them;
    without, they are the ``(n, length)`` class indices, with -100 on the
    tokens that do not count.

    Parameters
    ----------
    model : str or PreTrainedModel
        A model name or path, loaded with
        `AutoModelForTokenClassification.from_pretrained`, or a loaded
        model.
    num_labels : int, optional
        Number of classes; the number of `label_names` when not given, or
        read from a loaded model.
    label_names : sequence of str, optional
        The name of each class, by index, e.g. ``["O", "B-PER",
        "I-PER"]``. Also given to `from_pretrained` as ``id2label``.
    **kwargs
        Passed to `from_pretrained`.

    Examples
    --------
    >>> tags = ["O", "B-PER", "I-PER", "B-LOC", "I-LOC"]
    >>> model = TokenClassifier("bert-base-cased", label_names=tags)
    >>> trainer = Trainer(model, optimizer, metrics={"f1": SpanF1()})
    >>> trainer.fit(train_loader, valid_loader, epochs=3)
    """

    def __init__(
        self,
        model: str | PreTrainedModel,
        num_labels: int | None = None,
        *,
        label_names: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if label_names is not None:
            label_names = list(label_names)
            if num_labels is None:
                num_labels = len(label_names)
            if isinstance(model, str):
                kwargs.setdefault("id2label", dict(enumerate(label_names)))
                kwargs.setdefault(
                    "label2id", {n: i for i, n in enumerate(label_names)}
                )
        self.model = _load(
            AutoModelForTokenClassification, model, num_labels, kwargs
        )
        self.num_labels: int = self.model.config.num_labels
        if label_names is not None and len(label_names) != self.num_labels:
            msg = (
                f"got {len(label_names)} label names for a model of "
                f"{self.num_labels} classes"
            )
            raise ValueError(msg)
        self.label_names = label_names
        self._inputs = _input_names(self.model)

    def forward(self, **inputs: Any) -> torch.Tensor:
        """The ``(n, length, k)`` logits of the model's inputs."""
        return self.model(**inputs).logits

    def training_step(self, batch: Any) -> torch.Tensor:
        # the loss alone, without converting predictions to tags
        logits = self(**_inputs(batch, self._inputs))
        return _token_loss(logits, batch["labels"])

    def predict_step(self, batch: Any) -> dict[str, Any]:
        logits = self(**_inputs(batch, self._inputs))
        labels = batch.get("labels") if hasattr(batch, "get") else None
        if labels is not None:
            mask = labels != PAD_INDEX
        elif "attention_mask" in batch:
            mask = batch["attention_mask"].bool()
        else:
            mask = torch.ones(logits.shape[:2], dtype=torch.bool)
            mask = mask.to(logits.device)
        y_pred = logits.argmax(-1)
        outputs: dict[str, Any] = {"logits": logits, "mask": mask}
        if self.label_names is None:
            outputs["y_pred"] = y_pred.masked_fill(~mask, PAD_INDEX)
            if labels is not None:
                outputs["y_true"] = labels
        else:
            outputs["y_pred"] = self._tags(y_pred, mask)
            if labels is not None:
                outputs["y_true"] = self._tags(labels, mask)
        if labels is not None:
            outputs["loss"] = _token_loss(logits, labels)
        return outputs

    def _tags(self, ids: torch.Tensor, mask: torch.Tensor) -> list[list[str]]:
        """The tags of each sequence's tokens in `mask`."""
        names = self.label_names
        return [
            [names[i] for i in row[keep].tolist()]
            for row, keep in zip(ids.cpu(), mask.cpu())
        ]


def _load(
    auto: Any,
    model: str | PreTrainedModel,
    num_labels: int | None,
    kwargs: dict[str, Any],
) -> PreTrainedModel:
    """The model of a name or path, or a loaded model checked to have
    `num_labels` outputs."""
    if isinstance(model, str):
        if num_labels is not None:
            kwargs = {**kwargs, "num_labels": num_labels}
        return auto.from_pretrained(model, **kwargs)
    if kwargs:
        msg = (
            "from_pretrained arguments apply to a model name or path, not "
            f"to a loaded model: {', '.join(kwargs)}"
        )
        raise TypeError(msg)
    if num_labels is not None and model.config.num_labels != num_labels:
        msg = (
            f"expected a model with {num_labels} outputs, got one with "
            f"{model.config.num_labels}"
        )
        raise ValueError(msg)
    return model


def _input_names(model: torch.nn.Module) -> set[str] | None:
    """The batch entries the model's forward takes; None when it takes
    any keyword."""
    params = inspect.signature(model.forward).parameters.values()
    if any(p.kind is p.VAR_KEYWORD for p in params):
        return None
    return {p.name for p in params} - {"labels"}


def _inputs(batch: Any, names: set[str] | None) -> dict[str, Any]:
    """The entries of a batch the model takes, without the labels."""
    return {
        k: v
        for k, v in batch.items()
        if k != "labels" and (names is None or k in names)
    }


def _token_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """The mean cross-entropy over the tokens with a label."""
    return F.cross_entropy(
        logits.flatten(0, 1), labels.flatten(), ignore_index=PAD_INDEX
    )
