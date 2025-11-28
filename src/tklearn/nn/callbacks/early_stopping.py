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

if hasattr(np, "Inf"):
    Inf = np.Inf
else:
    Inf = np.inf


def get_monitor_op(mode: str, monitor: str) -> np.ufunc:
    if mode == "min":
        return np.less
    if mode == "max":
        return np.greater

    # Auto-detection logic
    if any(monitor.endswith(suffix) for suffix in NEG_METRICS_SUFFIX):
        return np.less
    if any(monitor.endswith(suffix) for suffix in POS_METRICS_SUFFIX):
        return np.greater

    raise ValueError(
        f"Could not infer the metric direction for {monitor}. Please specify mode='min' or 'max'."
    )


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
        # Initialize best based on mode (requires monitor_op to be resolvable)
        self.best = Inf if self.monitor_op == np.less else -Inf
        self.best_weights = None
        self.best_epoch = 0
        self.history = []

    @property
    def monitor(self) -> str:
        return self._monitor

    @monitor.setter
    def monitor(self, value: str):
        self._monitor = value
        self._monitor_op = None

    @property
    def mode(self) -> str:
        return self._mode

    @mode.setter
    def mode(self, value: str):
        if value not in {"auto", "min", "max"}:
            raise ValueError(
                f"Mode '{value}' is unknown, expected one of ('auto', 'min', 'max')"
            )
        self._mode = value
        self._monitor_op = None

    @property
    def monitor_op(self) -> np.ufunc:
        if getattr(self, "_monitor_op", None) is None:
            self._monitor_op = get_monitor_op(self.mode, self.monitor)
        return self._monitor_op

    def on_train_begin(self, logs=None):
        self.wait = 0
        self.stopped_epoch = 0
        self.best = Inf if self.monitor_op == np.less else -Inf
        self.best_weights = None
        self.best_epoch = 0
        self.history = []

    def _update_best(self, current, epoch):
        if self.verbose > 0:
            logger.debug(
                f"EarlyStopping: {self.monitor} improved from {self.best:.5f} "
                f"to {current:.5f} in epoch {epoch}"
            )
        self.best = current
        self.best_epoch = epoch

        if self.restore_best_weights:
            # NOTE: If 'tklearn.utils.copy' accepts 'device', keep it.
            # If using standard python copy, remove 'device="cpu"'.
            self.best_weights = copy.deepcopy(
                self.model.state_dict(), device="cpu"
            )

    def on_epoch_end(self, epoch: int, logs=None):
        current = self.get_monitor_value(logs)

        # Safety check for missing metrics or warm-up period
        if current is None or epoch < self.start_from_epoch:
            return

        # Fallback: Save initial weights if best_weights is empty
        # (e.g. if min_delta prevents the first epoch from registering as 'improvement')
        if self.restore_best_weights and self.best_weights is None:
            self.best_weights = copy.deepcopy(
                self.model.state_dict(), device="cpu"
            )

        self.wait += 1

        # Check if current result is an improvement over previous best
        if self._is_improvement(current, self.best):
            self._update_best(current, epoch)

            # Reset wait counter logic
            if self.baseline is None:
                self.wait = 0
            elif self._is_improvement(current, self.baseline):
                self.wait = 0
            return

        # Stopping logic
        if self.wait >= self.patience and epoch > 0:
            self.stopped_epoch = epoch
            self.model.stop_training = True

            if self.restore_best_weights and self.best_weights is not None:
                if self.verbose > 0:
                    logger.debug(
                        f"Restoring model weights from the end of the best epoch {self.best_epoch}."
                    )
                self.model.load_state_dict(self.best_weights, strict=True)

    def get_monitor_value(self, logs: Any):
        val = (logs or {}).get(self.monitor)
        # Optional: Handle NaN values which can break comparisons
        if val is not None and (np.isnan(val) or np.isinf(val)):
            # Treat NaN/Inf as "bad" result? Or let it crash?
            # usually safer to return None to skip logic.
            return None
        return val

    def _is_improvement(self, monitor_value, reference_value):
        if self.monitor_op == np.greater:
            return np.greater(monitor_value - self.min_delta, reference_value)
        return np.less(monitor_value + self.min_delta, reference_value)
