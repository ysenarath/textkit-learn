"""Types of data.

This module defines functions and classes for supporting data types when defining schema.
"""
import typing
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
import xarray as xr

___all__ = [
    'result_type',
    'from_data',
    'LogicalType',
    'logical_types',
]


def result_type(a, *args) -> str:
    """Get result type of given types.
    
    Parameters
    ----------
    a: str  or type
        Type to get result type from.
    args: str or type
        Other types to get result type from.

    Returns
    -------
    str
        Result type of given types.
    """
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
    """Get type from data
    
    Parameters
    ----------
    data: Any
        Data to get type from.

    Returns
    -------
    str
        Type of data.
    """
    if isinstance(data, Mapping):
        return 'object'
    elif isinstance(data, str):
        return 'str'
    elif isinstance(data, (np.ndarray, xr.DataArray, pd.Series, pd.DataFrame, Sequence)):
        return 'array'
    # infer type using numpy
    dtype = np.min_scalar_type(data)
    if isinstance(dtype, np.dtype):
        return dtype.name
    return type(data).__name__


# Logical types
class LogicalType(object):
    """Logical type of data."""

    def __init__(self, name, types=None, normalize=False):
        """Initialize logical type.
        
        Parameters
        ----------
        name: str
            Name of logical type.
        types: type or tuple[type]
            Types that this logical type can handle.
        normalize: bool
            Whether to normalize data after encoding or before decoding.
        """
        self._name = name
        if types is None:
            types = tuple()
        elif isinstance(types, type):
            types = (types,)
        self._types = types
        self._normalize = bool(normalize)
        self._ = False

    @property
    def normalize(self):
        """Whether to normalize data after encoding or before decoding.
        
        Returns
        -------
        bool
            Whether to normalize data after encoding or before decoding.
        """
        return self._normalize

    @property
    def name(self):
        """Name of logical type.
        
        Returns
        -------
        str
            Name of logical type.
        """
        return self._name

    def encode(self, data):
        """Encode data.
        
        Parameters
        ----------
        data: Any
            Data to encode.

        Returns
        -------
        Any
            Encoded data.
        """
        return data

    def decode(self, data):
        """Decode data.

        Parameters
        ----------
        data: Any
            Data to decode.

        Returns
        -------
        Any
            Decoded data.
        """
        return data

    def __repr__(self):
        """Get representation of logical type.
        
        Returns
        -------
        str
            String representation of logical type.        
        """
        return str(self._name)

    def issubclass(self, other):
        """Check whether given type is subclass of this logical type.
        
        Parameters
        ----------
        other: type
            Type to check.

        Returns
        -------
        bool
            Whether given type is subclass that is supported by this logical type.
        """
        if other is None:
            return False
        return issubclass(other, self._types)


class LogicalTypeManager(object):
    """Manager of logical types."""

    def __init__(self):
        """Initialize logical type manager."""
        self._ptypes = dict()

    def register(self, name, types, normalize=False):
        """Register logical type.
        
        Parameters
        ----------
        name: str
            Name of logical type.
        types: type or tuple[type]
            Types that this logical type can handle.
        normalize: bool
            Whether to normalize data after encoding or before decoding.

        Returns
        -------
        decorator : callable
            Callable decorator for registering logical type.                
        """
        def decorator(cls):
            pytype = cls(name, types, normalize)
            self._ptypes[name] = pytype
            return cls

        return decorator

    def __getitem__(self, item):
        """Get logical type by name or type.
        
        Parameters
        ----------
        item: str or type
            Name or type of logical type to get.

        Returns
        -------
        LogicalType
            Logical type that matches given name or type.
        """
        if isinstance(item, str):
            return self._ptypes[item]
        else:
            for name, pytype in self._ptypes.items():
                if pytype.issubclass(item):
                    return self[name]
        raise KeyError(item)

    def get(self, name, default=None):
        """Get logical type by name or type.

        Parameters
        ----------
        name: str or type
            Name or type of logical type to get.
        default: Any
            Default value to return if logical type is not found.

        Returns
        -------
        LogicalType
            Logical type that matches given name or type. If logical type is not found, return default value.
        """
        try:
            return self[name]
        except KeyError as ex:
            return default


# create global logical type manager.
logical_types = LogicalTypeManager()
