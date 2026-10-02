from __future__ import annotations

import math
import numbers
from typing import Literal

Mode = Literal["auto", "min", "max"]

# metric name endings whose direction can be inferred in "auto" mode
_MAXIMIZE_SUFFIXES = (
    "acc",
    "accuracy",
    "auc",
    "f1",
    "precision",
    "recall",
    "_score",
)
_MINIMIZE_SUFFIXES = ("loss", "error")


def resolve_mode(monitor: str, mode: Mode) -> Literal["min", "max"]:
    """Return whether `monitor` should be minimized or maximized."""
    if mode not in ("auto", "min", "max"):
        msg = f"mode must be one of 'auto', 'min' or 'max', got {mode!r}"
        raise ValueError(msg)
    if mode != "auto":
        return mode
    if monitor.endswith(_MINIMIZE_SUFFIXES):
        return "min"
    if monitor.endswith(_MAXIMIZE_SUFFIXES):
        return "max"
    msg = (
        f"could not infer whether to minimize or maximize {monitor!r}; "
        "pass mode='min' or mode='max'"
    )
    raise ValueError(msg)


def worst_value(mode: Literal["min", "max"]) -> float:
    return math.inf if mode == "min" else -math.inf


def is_improvement(
    current: float,
    reference: float,
    mode: Literal["min", "max"],
    min_delta: float = 0.0,
) -> bool:
    """Whether `current` beats `reference` by more than `min_delta`."""
    if mode == "max":
        return current - min_delta > reference
    return current + min_delta < reference


def is_valid_value(value: object) -> bool:
    return isinstance(value, numbers.Real) and math.isfinite(value)
