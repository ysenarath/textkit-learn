import typing
from collections.abc import Iterable, Sequence

import numcodecs
import numpy as np
import zarr

from tklearn.datasets import types
from tklearn.datasets.schema import Schema

__all__ = [
    'DocumentArray',
]


class DocumentArray(Sequence):
    IndexerType = typing.Union[int, slice, tuple[typing.Union[int, slice]]]

    def __init__(self, data: typing.Union[zarr.Group, zarr.Array] = None, schema: typing.Optional[Schema] = None):
        if data is None:
            data = zarr.group()
        self._data = data
        if schema is None:
            schema = Schema('array', items=Schema())
        self._schema = schema

    @property
    def fields(self) -> list[str]:
        keys = []
        if isinstance(self._data, zarr.Group):
            self._data.visitkeys(
                lambda x: keys.append(x) if isinstance(self._data[x], zarr.core.Array) else None
            )
        return keys

    def set_field(self, key, value):
        if isinstance(value, np.ndarray):
            arr = self._data.create_dataset(
                key, shape=value.shape,
                fill_value=None, dtype=value.dtype
            )
            arr[:] = value
            self._schema = Schema.from_data(value)
        raise NotImplementedError

    def get_field(self, key: str) -> 'DocumentArray':
        root = self._data[key]
        prop = self._schema[key]
        return DocumentArray(root, schema=prop)

    def __getitem__(self, index):
        index_iter = self._index_to_list(index)
        if index_iter is not None:
            # return Sequence of documents
            return [self[idx] for idx in index_iter]
        if not isinstance(index, int):
            raise TypeError(
                'document-array indices must be integers, sequence of integers or slices, not {}'.format(
                    type(index).__name__,
                )
            )
        if isinstance(self._data, zarr.Array):
            data = self._data[index]
        else:
            data = {}
            for key in self.fields:
                value = self.get_field(key)
                data[key] = value[index]
            # return documents
        return self._schema.items.denormalize(data)

    def __setitem__(self, index, value):
        index_iter = self._index_to_list(index)
        if index_iter is not None:
            iterable_value = False
            if isinstance(value, (Iterable, Sequence)):
                iterable_value = True
                input_size = len(value)
                index_size = len(index_iter)
                if input_size != index_size:
                    raise ValueError(
                        'could not broadcast input array from size {} into size {}'.format(
                            input_size,
                            index_size,
                        )
                    )
            for i, idx in enumerate(index_iter):
                self[idx] = value[i] if iterable_value else value
        elif not isinstance(index, int):
            raise TypeError(
                'document-array indices must be integers, sequence of integers or slices, not {}'.format(
                    type(index).__name__,
                )
            )
        elif isinstance(self._data, zarr.Array):
            self._data[index] = self._schema.items.normalize(value)
        else:
            doc, schema = self._schema.items.normalize(value, return_schema=True)
            for key, prop in schema.properties.items():
                value = doc.get(key)
                prop = schema.properties[key]
                if value is None and prop.type not in {'object', 'array'}:
                    value = zarr.empty(1, dtype=prop.type)[0]
                field = self.get_field(key)
                field[index] = value

    def append(self, doc: dict, dynamic=True):
        if dynamic:
            schema = Schema.from_data(doc)
            self._schema.items.update(schema)
        else:
            self._schema.items.validate(doc)
        self.resize(len(self) + 1)
        self[-1] = doc

    def __len__(self):
        if isinstance(self._data, zarr.Array):
            return self._data.shape[0]
        length = 0
        for key in self.fields:
            length = max(length, self.get_field(key).shape[0])
        return length

    @property
    def shape(self):
        return (len(self),)

    def resize(self, *args):
        if isinstance(self._data, zarr.Array):
            self._data.resize(*args)
        else:
            length = args[0]
            group = self._data
            _, schema = self._schema.normalize(None, return_schema=True, validate=False)
            for key, prop in schema.properties.items():
                if key not in group:
                    if prop.type == 'array' and prop.items.pytype is not None:
                        group.create_dataset(
                            key, shape=(0, *prop.items.shape),
                            dtype=prop.items.dtype
                        )
                    elif prop.type == 'object':
                        group.create_dataset(
                            key, shape=(0,),
                            fill_value=None,
                            dtype=object, object_codec=numcodecs.JSON()
                        )
                    elif prop.type == 'array':
                        group.create_dataset(
                            key, shape=(0,),
                            fill_value=None,
                            dtype=object, object_codec=numcodecs.JSON()
                        )
                    else:
                        group.create_dataset(
                            key, shape=(0,),
                            dtype=prop.type
                        )
                field = self.get_field(key)
                if not isinstance(field._data, zarr.Array):
                    continue
                shape = field._data.shape
                field.resize(length, *shape[1:])

    def _index_to_list(self, index):
        index_iter = None
        if isinstance(index, slice):
            index_iter = range(
                0 if index.start is None else index.start,
                len(self) if index.stop is None else index.stop,
                1 if index.step is None else index.step,
            )
        elif isinstance(index, (Iterable, Sequence)) and not isinstance(index, str):
            index_iter = index
        return index_iter


@types.pytypes.register('numpy.ndarray', types=np.ndarray)
class NumpyArrayStrategy(types.PyType):
    def transform(self, data):
        return data

    def inverse_transform(self, data):
        if isinstance(data, np.ndarray):
            return data
        raise TypeError('expected \'np.ndarray\', found {}'.format(type(data).__name__))
