from __future__ import annotations

import math
import numbers
import warnings
from collections.abc import Mapping
from typing import Any, Literal

from tklearn.nn.callbacks.base import Callback

__all__ = [
    "Mode",
    "MonitorCallback",
]

Mode = Literal["auto", "min", "max"]

# endings of log keys whose direction "auto" mode infers
_MIN_SUFFIXES = ("loss", "error", "mae", "mse", "rmse")
_MAX_SUFFIXES = (
    "acc",
    "accuracy",
    "auc",
    "auroc",
    "f1",
    "precision",
    "recall",
    "r2",
    "score",
    "pearson",
    "spearman",
    "correlation",
)


class MonitorCallback(Callback):
    """Base of callbacks that watch one value of the epoch logs.

    Parameters
    ----------
    monitor : str
        Key of the watched value in the logs, e.g. ``"valid_loss"``.
    mode : {"auto", "min", "max"}
        Whether lower or higher values are better. ``"auto"`` infers it
        from the key: ``*loss`` and ``*error`` are minimized, ``*acc``,
        ``*f1``, ``*auc``, ``*score``, ... are maximized.
    min_delta : float
        Smallest change that counts as an improvement.
    """

    def __init__(self, monitor: str, mode: Mode, min_delta: float) -> None:
        self.monitor = monitor
        self.mode = _resolve_mode(monitor, mode)
        self.min_delta = abs(min_delta)

    def _initial_best(self) -> float:
        return math.inf if self.mode == "min" else -math.inf

    def _is_improvement(self, current: float, reference: float) -> bool:
        """Whether `current` beats `reference` by more than `min_delta`.

        NaN never improves.
        """
        if self.mode == "min":
            return current + self.min_delta < reference
        return current - self.min_delta > reference

    def _get_monitor_value(self, logs: Mapping[str, Any]) -> float | None:
        """The watched value; None, with a warning, when it is missing."""
        value = logs.get(self.monitor)
        if value is None:
            msg = (
                f"{type(self).__name__} monitors {self.monitor!r}, which is "
                f"not in the logs; the logs have {', '.join(logs) or 'none'}"
            )
            warnings.warn(msg, stacklevel=2)
            return None
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            msg = (
                f"{type(self).__name__} monitors {self.monitor!r}, which "
                f"must be a number, got {value!r}"
            )
            raise TypeError(msg)
        return float(value)


def _resolve_mode(monitor: str, mode: Mode) -> Literal["min", "max"]:
    if mode not in ("auto", "min", "max"):
        msg = f"mode must be 'auto', 'min' or 'max', got {mode!r}"
        raise ValueError(msg)
    if mode != "auto":
        return mode
    if monitor.endswith(_MIN_SUFFIXES):
        return "min"
    if monitor.endswith(_MAX_SUFFIXES):
        return "max"
    msg = (
        f"cannot tell whether to minimize or maximize {monitor!r}; "
        "pass mode='min' or mode='max'"
    )
    raise ValueError(msg)
