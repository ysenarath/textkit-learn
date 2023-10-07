from functools import partial, wraps
import typing
from weakref import WeakValueDictionary
from urllib.parse import urlparse

import pyarrow as pa
from pyarrow import json
import datasets

from tklearn.datasets.dataset import Dataset, MapBatch

__all__ = [
    "DatasetLoader",
]


loaders = WeakValueDictionary()


class DatasetLoader(object):
    def __init__(self, func, name=None, batched=False):
        self.func = func
        self.name = name or func.__name__
        self.batched = batched
        loaders[self.name] = self

    def __call__(self, *args, **kwargs):
        return MapBatch(
            partial(self.func, *args, **kwargs), batched=self.batched
        ).to_dataset()


def register(
    func=None, *, name=None, batched=False
) -> typing.Union[DatasetLoader, partial]:
    if isinstance(func, str):
        # replace name with func
        return partial(register, name=func, batched=batched)
    return DatasetLoader(func, name=name, batched=batched)


@register("json", batched=True)
def read_json(path: str) -> pa.Table:
    """Read a Dataset from a stream of JSON data.."""
    yield json.read_json(path)


@register("hf", batched=False)
def read_hf_dataset(*args, **kwargs) -> pa.Table:
    """Read a Dataset from HuggingFace Datasets."""
    dataset = datasets.load_dataset(*args, **kwargs)
    for record in dataset:
        yield record


def load_dataset(__name__: typing.Optional[str] = None, *args, **kwargs) -> Dataset:
    """Load a Dataset by name."""
    if __name__ is None:
        path = None
        if len(args) > 0:
            path = args[0]
        path = str(kwargs.get("path", path))
        if path.endswith(".json"):
            __name__ = "json"
    if __name__ is None:
        raise ValueError("unable to determine loader name")
    if __name__ not in loaders:
        raise ValueError(f"no loader named {__name__}")
    loader = loaders[__name__]
    return loader(*args, **kwargs)
