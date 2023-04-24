import typing
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
import xarray as xr

___all__ = [
    'result_type',
    'from_data',
    'PyType',
    'get_pytype',
]


def result_type(a, *args) -> str:
    if not isinstance(a, str) and hasattr(a, 'type'):
        a = a.type
    if len(args) == 0:
        return a
    b = result_type(*args)
    if a is None:
        return b
    if not isinstance(a, str):
        raise TypeError('expect type to be \'str\' not \'{}\''.format(type(a).__name__))
    if a == b:
        return a
    if a in {'object', 'array'}:
        raise TypeError('type conflict found \'{}\', expected \'{}\''.format(b, a))
    dtype = np.result_type(a, b)
    # convert type to `str` that represents name
    if isinstance(dtype, np.dtype):
        dtype = dtype.name
    return dtype


def from_data(data: typing.Any) -> str:
    if isinstance(data, Mapping):
        return 'object'
    elif isinstance(data, str):
        return 'str'
    elif isinstance(data, (np.ndarray, xr.DataArray, pd.Series, pd.DataFrame, Sequence)):
        return 'array'
    dtype = np.min_scalar_type(data)
    if isinstance(dtype, np.dtype):
        return dtype.name
    return type(data).__name__


class PyType(object):
    def __init__(self, name, types=None, normalize=False):
        self._name = name
        if types is None:
            types = tuple()
        elif isinstance(types, type):
            types = (types,)
        self._types = types
        self._normalize = bool(normalize)

    @property
    def normalize(self):
        return self._normalize

    @property
    def name(self):
        return self._name

    def transform(self, data):
        return data

    def inverse_transform(self, data):
        return data

    def __repr__(self):
        return str(self._name)

    def issubclass(self, other):
        if other is None:
            return False
        return issubclass(other, self._types)


class PyTypeManager(object):
    def __init__(self):
        self._ptypes = dict()

    def register(self, name, types, normalize=False):
        def decorator(cls):
            pytype = cls(name, types, normalize)
            self._ptypes[name] = pytype

        return decorator

    def __getitem__(self, item):
        if isinstance(item, str):
            return self._ptypes[item]
        else:
            for name, pytype in self._ptypes.items():
                if pytype.issubclass(item):
                    return self[name]
        raise KeyError(item)

    def get(self, name, default=None):
        try:
            return self[name]
        except KeyError as ex:
            return default


pytypes = PyTypeManager()
