from __future__ import annotations

from typing import Any

from tklearn.metrics.base import MetricBase, MetricVariable

__all__ = [
    "StepsCounter",
]


class StepsCounter(MetricBase):
    _instance = None
    count = MetricVariable[int]()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def reset(self) -> None:
        self.count = 0

    def update(self, **kwargs: Any) -> None:
        self.count += 1

    def result(self) -> int:
        return self.count

    def __repr__(self) -> str:
        return f"StepsCounter(count={self.count})"
