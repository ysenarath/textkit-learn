"""Dataset loader module. 

This module provides a decorator for dataset loader functions.
"""
import functools

from tklearn.datasets import Dataset

__all__ = [
    'dataloader',
    'load_dataset',
]


class DatasetLoader(object):
    """A decorator for dataset loader functions."""
    loaders = {}  # A dictionary of dataset loaders.

    def __new__(cls, func=None, name=None):
        """Create a new dataset loader."""
        if name in cls.loaders:
            return cls.loaders[name]
        self = super(DatasetLoader, cls).__new__(cls)
        if name is not None:
            cls.loaders[name] = self
        return self

    def __init__(self, func=None, name=None):
        """Initialize a dataset loader.
        
        Parameters
        ----------
        func : callable
            The dataset loader function.
        name : str
            The name of the dataset.
        """
        self.name = name
        if func is not None:
            self.func = func
        elif not hasattr(self, 'func'):
            self.func = None

    def load(self, *args, **kwargs):
        """Load a dataset.
        
        Parameters
        ----------
        args : tuple
            The positional arguments to pass to the dataset loader.
        kwargs : dict
            The keyword arguments to pass to the dataset loader.

        Returns
        -------
        Dataset
            The loaded dataset.
        """
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
    """Load a dataset.

    Parameters
    ----------
    name : str
        The name of the dataset to load.
    args : tuple
        The positional arguments to pass to the dataset loader.
    kwargs : dict
        The keyword arguments to pass to the dataset loader.

    Returns
    -------
    Dataset
        The loaded dataset.
    """
    return DatasetLoader(name).load(*args, **kwargs)


def dataloader(name):
    """A decorator for dataset loader functions.

    Parameters
    ----------
    name : str
        The name of the dataset.
    """
    return functools.partial(DatasetLoader, name=name)
