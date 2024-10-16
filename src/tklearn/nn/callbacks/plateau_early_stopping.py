from typing import Any, Literal, Optional

import numpy as np

from tklearn import logging
from tklearn.nn.callbacks.base import Callback
from tklearn.utils import copy

__all__ = [
    "PlateauEarlyStopping",
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


class PlateauEarlyStopping(Callback):
    def __init__(
        self,
        monitor: str = "valid_loss",
        # min_delta: float = 0,
        patience: int = 0,
        verbose: int = 0,
        mode: Literal["auto", "min", "max"] = "auto",
        baseline: Optional[float] = None,
        restore_best_weights: bool = True,
        start_from_epoch: int = 0,
        # plateau related parameters
        plateau_epochs: int = 5,
        plateau_threshold: float = 1e-4,
    ) -> None:
        super().__init__()
        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.mode = mode
        self.baseline = baseline
        self.restore_best_weights = restore_best_weights
        self.start_from_epoch = start_from_epoch
        self.plateau_epochs = plateau_epochs
        self.plateau_threshold = plateau_threshold
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

    def get_monitor_value(self, logs: Any):
        return (logs or {}).get(self.monitor)

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

        self.history.append(current)

        self.wait += 1

        if self._is_improvement(current, self.best):
            self._update_best(current, epoch)
            self.wait = 0

        if self.wait >= self.patience and epoch > 0:
            self.stopped_epoch = epoch
            if self.restore_best_weights and self.best_weights is not None:
                if self.verbose > 0:
                    logger.debug(
                        f"Restoring model weights from the end of the best epoch: {self.best_epoch}."
                    )
                self.model.load_state_dict(self.best_weights, strict=True)
            self.model.stop_training = True

    def _is_improvement(self, monitor_value, reference_value):
        # monitor_value is the new value, reference_value is the old value
        # new value > old value if monitor_op is np.greater
        # new value < old value if monitor_op is np.less
        return self.monitor_op(monitor_value, reference_value)

    def _is_plateau(self):
        if len(self.history) < self.plateau_epochs:
            return False

        recent_values = np.array(self.history[-self.plateau_epochs :])

        # mean_value = np.mean(recent_values)
        # max_deviation = np.max(np.abs(np.array(recent_values) - mean_value))

        # return max_deviation < self.plateau_threshold

        slope, _ = np.polyfit(range(self.plateau_epochs), recent_values, 1)

        # return abs(slope) < self.plateau_threshold

        # Convert slope to angle in radians
        angle_radians = np.arctan(slope)

        if self.monitor_op == np.greater:
            # For maximization, we want a positive slope
            return angle_radians < self.plateau_threshold
        else:
            # For minimization, we want a negative slope
            return angle_radians > -self.plateau_threshold
