from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from opentelemetry import trace

from tklearn.logging import get_logger
from tklearn.nn.callbacks._monitor import Mode, MonitorCallback

if TYPE_CHECKING:
    from tklearn.nn.trainer import Trainer

__all__ = [
    "EarlyStopping",
]

logger = get_logger(__name__)


class EarlyStopping(MonitorCallback):
    """Stop training when a monitored value has stopped improving.

    After each epoch, the callback compares the monitored value with the
    best so far. Training stops once `patience` epochs in a row have not
    improved on it (at least one, when `patience` is 0).

    Parameters
    ----------
    monitor : str, default="valid_loss"
        Key of the watched value in the epoch logs.
    min_delta : float, default=0
        Smallest change that counts as an improvement.
    patience : int, default=0
        Epochs without improvement after which training stops.
    mode : {"auto", "min", "max"}, default="auto"
        Whether lower or higher values are better; ``"auto"`` infers it
        from the key (``*loss`` minimizes, ``*acc``, ``*f1``, ...
        maximize).
    baseline : float, optional
        A value to reach: training stops after `patience` epochs that do
        not improve on it, even if they improve on the best value.
    restore_best_weights : bool, default=False
        Load the model weights of the best epoch at the end of `fit`, when
        any epoch was monitored. The optimizer state is not restored.
    start_from_epoch : int, default=0
        Epochs to train before monitoring starts (a warm-up).
    verbose : bool, default=False
        Log when training stops and when weights are restored.

    Attributes
    ----------
    best : float
        Best monitored value of the current `fit` call.
    best_epoch : int
        Epoch of `best`.
    stopped_epoch : int or None
        Epoch after which this callback stopped training, if it did.
    wait : int
        Monitored epochs since the last improvement.

    Examples
    --------
    >>> early_stopping = EarlyStopping(
    ...     "valid_f1", patience=3, restore_best_weights=True
    ... )
    >>> trainer = Trainer(model, optimizer, callbacks=[early_stopping])
    """

    def __init__(
        self,
        monitor: str = "valid_loss",
        *,
        min_delta: float = 0.0,
        patience: int = 0,
        mode: Mode = "auto",
        baseline: float | None = None,
        restore_best_weights: bool = False,
        start_from_epoch: int = 0,
        verbose: bool = False,
    ) -> None:
        super().__init__(monitor, mode, min_delta)
        if patience < 0:
            msg = f"patience must be non-negative, got {patience}"
            raise ValueError(msg)
        self.patience = patience
        self.baseline = baseline
        self.restore_best_weights = restore_best_weights
        self.start_from_epoch = start_from_epoch
        self.verbose = verbose
        self._reset()

    def _reset(self) -> None:
        self.wait = 0
        self.stopped_epoch: int | None = None
        self.best = self._initial_best()
        self.best_epoch = 0
        self.best_weights: dict[str, torch.Tensor] | None = None

    def on_train_begin(self, trainer: Trainer) -> None:
        self._reset()

    def on_epoch_end(self, trainer: Trainer, logs: dict[str, Any]) -> None:
        if trainer.epoch < self.start_from_epoch:
            return
        current = self._get_monitor_value(logs)
        if current is None:
            return
        if self.restore_best_weights and self.best_weights is None:
            # restore the first monitored weights if no epoch improves
            self.best_weights = _copy_state(trainer.model)
            self.best_epoch = trainer.epoch
        self.wait += 1
        if self._is_improvement(current, self.best):
            self.best, self.best_epoch = current, trainer.epoch
            if self.restore_best_weights:
                self.best_weights = _copy_state(trainer.model)
            if self.baseline is None or self._is_improvement(
                current, self.baseline
            ):
                self.wait = 0
        # unlike Keras, this also stops when improving short of the baseline
        if self.wait >= max(self.patience, 1):
            self.stopped_epoch = trainer.epoch
            trainer.should_stop = True
            trace.get_current_span().add_event(
                "early_stopping",
                {
                    "monitor": self.monitor,
                    "best": self.best,
                    "best_epoch": self.best_epoch,
                    "wait": self.wait,
                },
            )

    def on_train_end(self, trainer: Trainer) -> None:
        verbose = self.verbose and trainer.accelerator.is_main_process
        if self.stopped_epoch is not None and verbose:
            # epochs are zero-based, but shown one-based like ProgbarLogger
            logger.info(
                f"Epoch {self.stopped_epoch + 1}: early stopping; "
                f"{self.monitor} did not improve for {self.wait} epochs"
            )
        if self.restore_best_weights and self.best_weights is not None:
            if verbose:
                logger.info(
                    "Restoring the model weights of epoch "
                    f"{self.best_epoch + 1}, the best with "
                    f"{self.monitor}={self.best:.4g}"
                )
            trainer.model.load_state_dict(self.best_weights)
            trace.get_current_span().add_event(
                "restore_best_weights",
                {
                    "epoch": self.best_epoch,
                    "monitor": self.monitor,
                    "best": self.best,
                },
            )


def _copy_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """A copy of the model's state on the CPU."""
    return {
        key: value.to("cpu", copy=True)
        for key, value in model.state_dict().items()
    }
