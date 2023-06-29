"""Defines base-classes for types of datasets."""
from typing import Any
import typing


__all__ = [
    'get_dataset_loader',
    'register',
]


datasets = {}
dataset_aliases = {}


class DatasetLoader(object):
    def __init__(self, name, loader, *, aliases=None, splits=None, citation=None, description=None) -> None:
        self._name = name
        self._loader = loader
        self._citation = citation
        self._description = description
        self._splits = tuple(splits) if splits is not None else tuple()
        self._aliases = tuple(aliases) if aliases is not None else tuple()

    @property
    def name(self):
        return self._name

    @property
    def aliases(self):
        return self._aliases

    @property
    def splits(self):
        return self._splits

    @property
    def citation(self):
        return self._citation

    @property
    def description(self):
        return self._description

    def load(self, *args: Any, **kwds: Any) -> Any:
        return self._loader(*args, **kwds)

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.load(*args, **kwds)


def register(name, aliases=None, splits=None):
    def decorator(loader: typing.Callable):
        datasets[name] = DatasetLoader(
            name, loader=loader, aliases=aliases, splits=splits)
        dataset_aliases[name] = name
        if aliases is not None:
            for alias in aliases:
                dataset_aliases[alias] = name
        return loader
    return decorator


def get_dataset_loader(name):
    return datasets[dataset_aliases[name]]
