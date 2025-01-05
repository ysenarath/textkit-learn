from __future__ import annotations

import functools
from dataclasses import MISSING
from itertools import islice
from typing import (
    Any,
    Callable,
    Generic,
    Iterable,
    List,
    Mapping,
    Optional,
    TypeVar,
)
from weakref import WeakKeyDictionary

import pandas as pd
import torch

from tklearn.utils import hashing
from tklearn.utils.cache import FileCache

T = TypeVar("T")
I = TypeVar("I")  # noqa: E741


def validate(results: Any) -> Iterable[Any]:
    if isinstance(results, pd.DataFrame):
        # Pandas DataFrame
        return results.to_dict(orient="records")
    elif isinstance(results, Mapping):
        # assume that mapping is a dict of lists
        key = next(iter(results.keys()))
        N = len(results[key])
        expected = [{} for _ in range(N)]
        for key in results.keys():
            for i in range(N):
                expected[i][key] = results[key][i]
        return expected
    elif torch.is_tensor(results):
        # PyTorch tensors
        return results
    elif pd.api.types.is_list_like(results, allow_sets=False):
        # lists, tuples, NumPy arrays, Pandas Series
        return results
    raise TypeError(f"invalid return type: {results.__class__.__name__}")


class Future(Generic[T]):
    def __init__(self, task: LazyTask, fingerprint: str):
        self.task = task
        self.fingerprint = fingerprint
        self._result = MISSING

    def result(self) -> T:
        if self._result is MISSING:
            self._result = self.task.compute(self)
        return self._result


class Task(Generic[I, T]):
    def __init__(self, task: Callable[[I], T]):
        self.task = task

    def __call__(self, obj: I) -> T:
        return self.task(obj)


class LazyTask(Task[I, T]):
    def __init__(
        self,
        task: Callable[[List[I]], T],
        batched: bool = False,
        batch_size: Optional[int] = None,
        cached: bool = False,
    ):
        self.task = task
        self.batched = batched
        self.batch_size = batch_size
        self.buffer = WeakKeyDictionary()
        self.results = WeakKeyDictionary()
        self.cache = FileCache() if cached else None
        self.fingerprint = hashing.hash(self)

    def process(self):
        if self.batched:
            if self.batch_size is None:
                batch_size = 32
            else:
                batch_size = self.batch_size
            while self.buffer:
                futures = list(islice(self.buffer, batch_size))
                buffer = []
                for future in futures:
                    obj = self.buffer.pop(future)
                    buffer.append(obj)
                results = validate(self.task(buffer))
                for future, result in zip(futures, results):
                    self.results[future] = result
        else:
            if self.batch_size:
                raise ValueError(
                    "batch_size must be None for non-batched tasks"
                )
            for future, obj in self.buffer.items():
                self.results[future] = self.task(obj)

    def compute(self, future: Future) -> T:
        try:
            if not self.cache:
                # proceed to computation
                raise FileNotFoundError
            result = self.cache.load(future.fingerprint)
            self.buffer.pop(future)
        except FileNotFoundError:
            if future not in self.results:
                self.process()
            result = self.results.pop(future)
            if self.cache:
                self.cache.dump(
                    result,
                    fingerprint=future.fingerprint,
                )
        return result

    def __call__(self, obj: I) -> Future[T]:
        # take fingerprint of obj and task
        if self.cache:
            fingerprint = hashing.hash(self.fingerprint, obj)
        else:
            fingerprint = None
        future = Future(self, fingerprint=fingerprint)
        self.buffer[future] = obj
        return future

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state.pop("buffer", None)
        state.pop("results", None)
        state.pop("cache", None)
        state.pop("fingerprint", None)
        state["cached"] = self.cache is not None
        return state

    def __setstate__(self, state: dict):
        cached = state.pop("cached")
        self.__dict__.update(state)
        self.buffer = WeakKeyDictionary()
        self.results = WeakKeyDictionary()
        self.cache = FileCache() if cached else None
        self.fingerprint = hashing.hash(self)


def task(
    func: Any | None = None,
    lazy: bool = True,
    batched: bool = True,
    cached: bool = False,
) -> Any:
    if func is not None:
        return task()(func)

    def decorator(func: Any) -> Any:
        if batched:
            if not lazy:
                raise ValueError("batched tasks must be lazy")

            def wrapped(batch_size: int = 32) -> LazyTask:
                return LazyTask(
                    func, batched=True, batch_size=batch_size, cached=cached
                )

        elif lazy:

            def wrapped() -> LazyTask:
                return LazyTask(func, cached=cached)

        else:

            def wrapped() -> Task:
                return Task(func)

        assigned = ("__module__", "__name__", "__qualname__")
        return functools.wraps(func, assigned=assigned)(wrapped)

    return decorator
