import functools

from tklearn.datasets import Dataset

__all__ = [
    'dataloader',
    'load_dataset',
]


class DatasetLoader(object):
    loaders = {}

    def __new__(cls, func=None, name=None):
        if name in cls.loaders:
            return cls.loaders[name]
        self = super(DatasetLoader, cls).__new__(cls)
        if name is not None:
            cls.loaders[name] = self
        return self

    def __init__(self, func=None, name=None):
        self.name = name
        if func is not None:
            self.func = func
        elif not hasattr(self, 'func'):
            self.func = None

    def load(self, *args, **kwargs):
        func = self.func
        if func is None:
            docs = None
        else:
            docs = func(*args, **kwargs)
        dataset = Dataset()
        for doc in docs:
            dataset.append(doc)
        return dataset


def load_dataset(name, *args, **kwargs):
    return DatasetLoader(name).load(*args, **kwargs)


def dataloader(name):
    return functools.partial(DatasetLoader, name=name)
