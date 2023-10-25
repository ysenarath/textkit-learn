from __future__ import annotations
from typing import Callable, List
import functools

__all__ = [
    "Pipeline",
]


class Pipeline(object):
    def __init__(self, pipeline: List[Callable]) -> None:
        self.pipeline = pipeline

    def forward(self, *args, **kwargs):
        output = None
        for pipe in self.pipeline:
            output = pipe(output)
        return output

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def pipe(self, other: Callable) -> Pipeline:
        pipeline = self.pipeline.copy() + [other]
        return Pipeline(pipeline)


def pipeline(func: Callable):
    self = Pipeline()

    def forward(self, *args, **kwargs):
        return func(*args, **kwargs)

    self.forward = forward

    functools.update_wrapper(self, func)
    return self
