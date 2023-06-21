"""Mapping of data to a document.

This module provides a mapping of data to a document.

Example
-------
>>> from tklearn.datasets.mapping import Mapping
>>> mapping = Mapping.from_data({'a': 1, 'b': 2})
"""
import typing
from collections.abc import Iterable, Mapping as DictLike

import numpy as np
import pandas as pd
from pydantic import BaseModel

from tklearn.datasets.collection import Collection
from tklearn.datasets.document import Document

__all__ = [
    'Mapping',
]


class Mapping(BaseModel):
    """Mapping of data to a document.

    Parameters
    ----------
    type : str
        Type of the mapping.
    properties : dict[str, Mapping], optional
        Properties of the mapping, by default None
    required : list[str], optional
        Required properties of the mapping, by default None
    items : Mapping, optional
        Items of the mapping, by default None
    """
    type: str
    properties: typing.Optional[dict[str, 'Mapping']] = None
    required: typing.Optional[list[str]] = None
    items: typing.Optional['Mapping'] = None
    default: typing.Optional[typing.Any] = None

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) \
            -> 'Mapping':
        """Create a mapping from a dataframe.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe to be mapped.

        Returns
        -------
        Mapping
            Mapping of the dataframe.
        """
        props = {}
        required = []
        for k, v in df.dtypes.items():
            if v.name != 'object':
                props[k] = cls(type=v.name)
                required.append(k)
                continue
            # type of the column is 'object'
            item, i = None, 0
            while item is None:
                item = df.iloc[i][k]
                i += 1
            props[k] = cls.from_data(item)
        return cls(
            type='array',
            items=cls(type='object', properties=props, required=required)
        )

    @classmethod
    def from_data(cls, data):
        """Create a mapping from data.

        Parameters
        ----------
        data : Any
            Data to be mapped.

        Returns
        -------
        Mapping
            Mapping of the data.
        """
        if isinstance(data, pd.DataFrame):
            return cls.from_dataframe(data)
        # if isinstance(data, pd.Series):
        #     return cls.from_series(data)
        if isinstance(data, DictLike):
            props = {}
            required = []
            for k, v in data.items():
                props[k] = cls.from_data(v)
                if v is not None:
                    required.append(k)
            return cls(type='object', properties=props, required=required)
        elif isinstance(data, Iterable) and not isinstance(data, str):
            mappings = [cls.from_data(d) for d in data]
            return cls(type='array', items=cls.merge(mappings))
        elif isinstance(data, (int, float, bool, str)):
            return cls(type=type(data).__name__)
        elif data is None:
            return cls(type='null')
        dtype = np.min_scalar_type(data).name
        return cls(type=dtype)

    @classmethod
    def merge(cls, mappings: list['Mapping']) -> 'Mapping':
        """Merge multiple mappings into one.

        Parameters
        ----------
        mappings : list[Mapping]
            List of mappings to be merged.

        Returns
        -------
        Mapping
            Merged mapping.
        """
        if len(mappings) == 0:
            raise ValueError('mappings cannot be empty')
        elif len(mappings) == 1:
            return mappings[0]
        elif mappings[0].type == 'object':
            mapping = cls(type='object', properties={}, required=[])
            for s in mappings:
                # merge properties
                for k, v in s.properties.items():
                    if k not in mapping.properties:
                        mapping.properties[k] = v
                        continue
                    mapping.properties[k] = cls.merge(
                        [mapping.properties[k], v]
                    )
                # merge required
                rc = set(mapping.required) \
                    .intersection(set(s.required))
                mapping.required = list(rc)
            return mapping
        elif mappings[0].type == 'array':
            mapping = cls(type='array')
            # merge items
            mapping.items = cls.merge([s.items for s in mappings])
            return mapping
        # use numpy to merge types
        dtype = np.result_type(
            *[s.type for s in mappings if s.type]
        )
        return cls(type=dtype.name)

    def map_object(self, data) -> Document:
        """Map data to a document.

        Parameters
        ----------
        data : DictLike
            Data to be mapped.

        Returns
        -------
        Document
            Mapped document.
        """
        if self.type != 'object':
            raise ValueError('mapping must be an object')
        if not isinstance(data, DictLike):
            raise ValueError('data must be a dict')
        if self.properties is None:
            raise ValueError('properties must be defined')
        if not set(self.required).issubset(set(data.keys())):
            raise ValueError('required keys not found')
        props = {k: v.default for k, v in self.properties.items()}
        for k, v in data.items():
            if not self.properties or k not in self.properties:
                props[k] = v
                continue
            props[k] = self.properties[k].map(v)
        return Document(id=data.get('id', None), props=props)

    def map_array(self, data, offset=None, limit=None) \
            -> typing.Union[Collection, Iterable]:
        """Map data to a collection.

        Parameters
        ----------
        data : Iterable
            Data to be mapped.

        Returns
        -------
        Collection
            Mapped collection.
        """
        if self.type != 'array':
            raise ValueError('mapping must be an array')
        if not isinstance(data, Iterable) or isinstance(data, str):
            raise ValueError('data must be iterable')
        if self.items is None:
            raise ValueError('items must be defined')
        items = []
        is_collection = True
        if isinstance(data, pd.DataFrame):
            data = data.to_dict(orient='records')
        for d in data:
            if limit is not None and limit > 0:
                break
            d = self.items.map(d)
            if not isinstance(d, (Collection, Document)):
                is_collection = False
                break
            items.append(d)
            if limit is not None:
                limit -= 1
        if is_collection:
            return Collection(items=items, offset=offset)
        return data

    def map(self, data: typing.Any, **kwargs):
        """Map data to a document or collection.

        Parameters
        ----------
        data : Any
            Data to be mapped.

        Returns
        -------
        Document or Collection
            Mapped document or collection.
        """
        if self.type == 'object':
            return self.map_object(data, **kwargs)
        elif self.type == 'array':
            return self.map_array(data, **kwargs)
        return data
