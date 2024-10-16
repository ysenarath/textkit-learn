from typing import Any, Literal, Optional

import numpy as np

from tklearn import logging
from tklearn.nn.callbacks.base import Callback
from tklearn.utils import copy

__all__ = [
    "EarlyStopping",
]

logger = logging.get_logger(__name__)

POS_METRICS_SUFFIX = ["acc", "accuracy", "auc", "_score"]
NEG_METRICS_SUFFIX = ["loss", "error"]


def get_monitor_op(mode: str, monitor: str) -> np.ufunc:
    # give preference to the mode
    if mode == "min":
        return np.less
    if mode == "max":
        return np.greater
    if any(map(monitor.endswith, NEG_METRICS_SUFFIX)):
        return np.less
    if any(map(monitor.endswith, POS_METRICS_SUFFIX)):
        return np.greater
    msg = f"could not infer the metric direction for {monitor}."
    raise ValueError(msg)


class EarlyStopping(Callback):
    def __init__(
        self,
        monitor: str = "valid_loss",
        min_delta: float = 0,
        patience: int = 0,
        verbose: int = 0,
        mode: Literal["auto", "min", "max"] = "auto",
        baseline: Optional[float] = None,
        restore_best_weights: bool = True,
        start_from_epoch: int = 0,
    ) -> None:
        super().__init__()
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.verbose = verbose
        self.mode = mode
        self.baseline = baseline
        self.restore_best_weights = restore_best_weights
        self.start_from_epoch = start_from_epoch
        # internal variables
        self.wait = 0
        self.stopped_epoch = 0
        self.best = np.Inf if self.monitor_op == np.less else -np.Inf
        self.best_weights = None
        self.best_epoch = 0
        self.history = []

    @property
    def monitor_op(self) -> np.ufunc:
        if getattr(self, "_monitor_op", None) is None:
            self._monitor_op = get_monitor_op(self.mode, self.monitor)
        return self._monitor_op

    @property
    def mode(self) -> str:
        return self._mode

    @mode.setter
    def mode(self, mode):
        if mode not in {"auto", "min", "max"}:
            msg = (
                f"mode '{mode}' is unknown, "
                'expected one of ("auto", "min", "max")'
            )
            raise ValueError(msg)
        self._mode = mode
        self._monitor_op = None

    @property
    def monitor(self) -> str:
        return self._monitor

    @monitor.setter
    def monitor(self, mode):
        self._monitor = mode
        self._monitor_op = None

    def on_train_begin(self, logs=None):
        self.wait = 0
        self.stopped_epoch = 0
        self.best = np.Inf if self.monitor_op == np.less else -np.Inf
        self.best_weights = None
        self.best_epoch = 0
        self.history = []

    def _update_best(self, current, epoch):
        if self.verbose > 0:
            msg = (
                f"PlateauEarlyStopping: {self.monitor} "
                f"improved from {self.best:.5f} "
                f"to {current:.5f} in epoch {epoch}"
            )
            logger.debug(msg)
        self.best = current
        self.best_epoch = epoch
        if self.restore_best_weights:
            self.best_weights = copy.deepcopy(
                self.model.state_dict(), device="cpu"
            )

    def on_epoch_end(self, epoch: int, logs=None):
        current = self.get_monitor_value(logs)

        if current is None or epoch < self.start_from_epoch:
            # If no monitor value exists or still in initial warm-up stage.
            return

        if self.restore_best_weights and self.best_weights is None:
            # Restore the weights after first epoch if no progress is ever made.
            self.best_weights = copy.deepcopy(
                self.model.state_dict(), device="cpu"
            )

        self.wait += 1
        if self._is_improvement(current, self.best):
            self._update_best(current, epoch)
            # Only restart wait if we beat both the baseline and our previous
            # best.
            if self.baseline is None or self._is_improvement(
                current, self.baseline
            ):
                self.wait = 0
            return

        if self.wait >= self.patience and epoch > 0:
            self.stopped_epoch = epoch
            if self.restore_best_weights and self.best_weights is not None:
                if self.verbose > 0:
                    pass
                self.model.load_state_dict(self.best_weights, strict=True)
            self.model.stop_training = True

    def get_monitor_value(self, logs: Any):
        return (logs or {}).get(self.monitor)

    def _is_improvement(self, monitor_value, reference_value):
        # monitor_value is the new value, reference_value is the old (best) value
        if self.monitor_op == np.greater:
            # new value > old value + min delta
            return np.greater(monitor_value - self.min_delta, reference_value)
        # new value < old value - min delta
        return np.less(monitor_value + self.min_delta, reference_value)
