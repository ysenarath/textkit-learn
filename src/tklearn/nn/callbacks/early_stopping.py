from __future__ import annotations

from typing import Any

from tklearn import logging
from tklearn.nn.callbacks._monitor import (
    Mode,
    is_improvement,
    is_valid_value,
    resolve_mode,
    worst_value,
)
from tklearn.nn.callbacks.base import Callback
from tklearn.utils import copy

__all__ = [
    "EarlyStopping",
]

logger = logging.get_logger(__name__)


class EarlyStopping(Callback):
    """Stop training when a monitored value stops improving.

    Parameters
    ----------
    monitor : str, default="valid_loss"
        Key in the epoch logs to watch.
    min_delta : float, default=0
        Smallest change that counts as an improvement.
    patience : int, default=0
        Epochs without improvement before stopping.
    verbose : int, default=0
        Log decisions at debug level when > 0.
    mode : {"auto", "min", "max"}, default="auto"
        Whether lower or higher is better; "auto" infers it from the name
        (``*loss``/``*error`` minimize; ``*acc``, ``*auc``, ``*f1``, ... maximize).
    baseline : float, optional
        Only reset the patience counter for values that also beat this.
    restore_best_weights : bool, default=True
        Restore the weights of the best epoch when stopping.
    start_from_epoch : int, default=0
        Ignore epochs before this one (warm-up).

    Attributes
    ----------
    best : float
        Best monitored value so far.
    best_epoch : int
        Epoch of `best`.
    stopped_epoch : int
        Epoch at which training was stopped, or 0.
    """

    def __init__(
        self,
        monitor: str = "valid_loss",
        min_delta: float = 0.0,
        patience: int = 0,
        verbose: int = 0,
        mode: Mode = "auto",
        baseline: float | None = None,
        restore_best_weights: bool = True,
        start_from_epoch: int = 0,
    ) -> None:
        super().__init__()
        self.monitor = monitor
        self.mode = resolve_mode(monitor, mode)
        self.min_delta = abs(min_delta)
        self.patience = patience
        self.verbose = verbose
        self.baseline = baseline
        self.restore_best_weights = restore_best_weights
        self.start_from_epoch = start_from_epoch
        self._reset()

    def _reset(self) -> None:
        self.wait = 0
        self.stopped_epoch = 0
        self.best = worst_value(self.mode)
        self.best_epoch = 0
        self.best_weights = None

    def _log(self, msg: str) -> None:
        if self.verbose > 0:
            logger.debug(f"EarlyStopping: {msg}")

    def on_train_begin(self, logs: dict[str, Any] | None = None) -> None:
        self._reset()

    def on_epoch_end(
        self, epoch: int, logs: dict[str, Any] | None = None
    ) -> None:
        current = (logs or {}).get(self.monitor)
        if not is_valid_value(current) or epoch < self.start_from_epoch:
            return
        if self.restore_best_weights and self.best_weights is None:
            # keep the first weights in case no epoch ever improves
            self.best_weights = copy.deepcopy(
                self.model.state_dict(), device="cpu"
            )
        self.wait += 1
        if is_improvement(current, self.best, self.mode, self.min_delta):
            self._log(
                f"{self.monitor} improved from {self.best:.5f} to "
                f"{current:.5f} at epoch {epoch}"
            )
            self.best, self.best_epoch = current, epoch
            if self.restore_best_weights:
                self.best_weights = copy.deepcopy(
                    self.model.state_dict(), device="cpu"
                )
            if self.baseline is None or is_improvement(
                current, self.baseline, self.mode
            ):
                self.wait = 0
            return
        if self.wait >= self.patience and epoch > 0:
            self.stopped_epoch = epoch
            if self.trainer is not None:
                self.trainer.stop_training = True
            self._log(
                f"stopping at epoch {epoch}: no improvement in "
                f"{self.monitor} for {self.wait} epochs"
            )
            if self.restore_best_weights and self.best_weights is not None:
                self._log(f"restoring weights from epoch {self.best_epoch}")
                self.model.load_state_dict(self.best_weights, strict=True)
