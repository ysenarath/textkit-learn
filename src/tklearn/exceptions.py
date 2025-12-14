from collections.abc import Sequence
from typing import Any

__all__ = [
    "UnexpectedValueError",
]


class UnexpectedValueError(ValueError):
    def __init__(self, got: Any, expected: Any):
        if (
            isinstance(expected, Sequence) and not isinstance(expected, str)
        ) or (isinstance(expected, str) and "," in expected):
            expected_str = ", ".join(map(str, expected))
            msg = f"got {got}, expected one of [{expected_str}]"
        else:
            expected_str = str(expected)
            msg = f"got {got}, expected {expected_str}"
        super().__init__(msg)
