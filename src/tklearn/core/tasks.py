from __future__ import annotations

import functools
from dataclasses import MISSING
from typing import Any, Callable, Mapping, TypeVar
from weakref import WeakKeyDictionary

from typing_extensions import ParamSpec

P = ParamSpec("P")
T = TypeVar("T")


class Future:
    def __init__(self, task: LazyTask):
        self.task = task
        self._result = MISSING

    def result(self) -> T:
        if self._result is MISSING:
            self._result = self.task.compute(self)
        return self._result


class LazyTask:
    def __init__(self, task: Callable, buffer_size: int = 32):
        self.task = task
        self.buffer_size = buffer_size
        self.buffer = WeakKeyDictionary()
        self.results = WeakKeyDictionary()

    def process(self):
        futures = list(self.buffer.keys())
        buffer = []
        for future in futures:
            obj = self.buffer.pop(future)
            buffer.append(obj)
        results = self.task(buffer)
        if isinstance(results, Mapping):
            key = next(iter(results.keys()))
            N = len(results[key])
            expected = [{} for _ in range(N)]
            for key in results.keys():
                for i in range(N):
                    expected[i][key] = results[key][i]
            results = expected
        for future, result in zip(futures, results):
            self.results[future] = result

    def compute(self, future: Future) -> Any:
        if future not in self.results:
            self.process()
        return self.results.pop(future)

    def __call__(self, obj: Any) -> Future:
        future = Future(self)
        self.buffer[future] = obj
        if len(self.buffer) >= self.buffer_size:
            self.process()
        return future


def task(task: Callable) -> Callable[[], LazyTask]:
    @functools.wraps(task)
    def wrapped(buffer_size: int = 32):
        return LazyTask(task, buffer_size=buffer_size)

    return wrapped
