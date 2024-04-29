import copy
from typing import Optional

import numpy as np

from tklearn.nn.callbacks.base import Callback
from tklearn.nn.utils.logging import get_logger

__all__ = [
    "EarlyStopping",
]

logger = get_logger(__name__)


class EarlyStopping(Callback):
    def __init__(
        self,
        monitor: str = "valid_loss",
        min_delta: float = 0,
        patience: int = 0,
        verbose: int = 0,
        mode: str = "auto",
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
        # {min, max, auto}
        self.wait = 0
        self.stopped_epoch = 0
        self.best = np.Inf if self.monitor_op == np.less else -np.Inf
        self.best_weights = None
        self.best_epoch = 0

    @property
    def mode(self) -> str:
        return self._mode

    @mode.setter
    def mode(self, mode):
        if mode not in {"auto", "min", "max"}:
            msg = f"mode {mode} is unknown, " 'expected one of ("auto", "min", "max")'
            raise ValueError(msg)
        self._mode = mode
        if mode == "min":
            self.monitor_op = np.less
        elif mode == "max":
            self.monitor_op = np.greater
        elif (
            self.monitor.endswith("acc")
            or self.monitor.endswith("auc")
            or self.monitor.endswith("_score")
        ):
            self.monitor_op = np.greater
        elif self.monitor.endswith("loss") or self.monitor.endswith("error"):
            self.monitor_op = np.less
        else:
            msg = f"could not infer the metric direction for {self.monitor}."
            raise ValueError(msg)
        if self.monitor_op == np.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1

    def on_train_begin(self, logs=None):
        self.wait = 0
        self.stopped_epoch = 0
        self.best = np.Inf if self.monitor_op == np.less else -np.Inf
        self.best_weights = None
        self.best_epoch = 0

    def on_epoch_end(self, epoch: int, logs=None):
        current = self.get_monitor_value(logs)
        if current is None or epoch < self.start_from_epoch:
            # If no monitor value exists or still in initial warm-up stage.
            return
        if self.restore_best_weights and self.best_weights is None:
            # Restore the weights after first epoch if no progress is ever made.
            self.best_weights = copy.deepcopy(self.model.state_dict())
        self.wait += 1
        if self._is_improvement(current, self.best):
            if self.verbose > 0:
                msg = (
                    f"EarlyStopping: {self.monitor} "
                    f"improved from {self.best:.5f} "
                    f"to {current:.5f} in epoch {epoch}"
                )
                logger.debug(msg)
            self.best = current
            self.best_epoch = epoch
            if self.restore_best_weights:
                self.best_weights = copy.deepcopy(self.model.state_dict())
            # Only restart wait if we beat both the baseline and our previous
            # best.
            if self.baseline is None or self._is_improvement(current, self.baseline):
                self.wait = 0
            return
        # Only check after the first epoch.
        if self.wait >= self.patience and epoch > 0:
            self.stopped_epoch = epoch
            if self.restore_best_weights and self.best_weights is not None:
                if self.verbose > 0:
                    pass
                self.model.load_state_dict(self.best_weights)
            self.model.stop_training = True

    def get_monitor_value(self, logs):
        logs = logs or {}
        monitor_value = logs.get(self.monitor)
        if monitor_value is None:
            pass
        return monitor_value

    def _is_improvement(self, monitor_value, reference_value):
        return self.monitor_op(monitor_value - self.min_delta, reference_value)
