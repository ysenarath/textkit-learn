import typing
from collections.abc import MutableSequence, Iterable, Sequence

import numcodecs
import numpy as np
import xarray as xr
import zarr

from tklearn.datasets import types
from tklearn.datasets.schema import Schema

__all__ = [
    'BaseArray',
    'DataArray',
    'DocumentArray',
]


class BaseArray(MutableSequence):
    def __setitem__(self, index, value):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

    def __delitem__(self, index):
        raise NotImplementedError

    def insert(self, index: int, value: typing.Any) -> None:
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    @property
    def dtype(self):
        raise NotImplementedError

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def shape(self):
        raise NotImplementedError

    def resize(self, *args):
        raise NotImplementedError

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


class DataArray(BaseArray):
    IndexerType = typing.Union[int, slice, tuple[typing.Union[int, slice]]]

    def __init__(self, array: zarr.Array, schema: typing.Optional[Schema] = None):
        self._array = array
        self._schema = schema

    @property
    def dtype(self):
        return self._array.dtype

    @property
    def shape(self):
        return self._array.shape

    def __getitem__(self, index: IndexerType) -> typing.Any:
        if not isinstance(index, int):
            raise NotImplementedError
        arr = self._array[index]  # type: np.ndarray
        # if np.isscalar(arr):
        #     return arr
        # return DataArray(arr, schema=self._schema.items)
        return self._schema.items.denormalize(arr)

    def __setitem__(self, index: IndexerType, value: typing.Any) -> None:
        if not isinstance(index, int):
            raise NotImplementedError
        self._array[index] = self._schema.items.normalize(value)

    def __len__(self) -> int:
        return len(self._array)

    def resize(self, *args):
        if not isinstance(self._array, zarr.Array):
            raise NotImplementedError
        self._array.resize(*args)

    def to_xarray(self) -> xr.DataArray:
        arr = self._array[:]
        return xr.DataArray(arr)

    def to_numpy(self) -> np.ndarray:
        return self._array[:]


class DocumentArray(BaseArray):
    def __init__(self, group: zarr.Group = None, schema: typing.Optional[Schema] = None):
        if group is None:
            group = zarr.group()
        self._group = group
        if schema is None:
            schema = Schema('array', items=Schema())
        self._schema = schema

    @property
    def fields(self) -> list[str]:
        keys = []
        self._group.visitkeys(lambda x: keys.append(x) if isinstance(self._group[x], zarr.core.Array) else None)
        return keys

    def set_field(self, key, value):
        if isinstance(value, np.ndarray):
            arr = self._group.create_dataset(
                key, shape=value.shape,
                fill_value=None, dtype=value.dtype
            )
            arr[:] = value
            self._schema = Schema.from_data(value)
        raise NotImplementedError

    def get_field(self, key: str) -> BaseArray:
        root = self._group[key]
        prop = self._schema[key]
        if isinstance(root, zarr.core.Array):
            return DataArray(root, schema=prop)
        # result is a subgroup of fields => it is a document-array
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
        length = 0
        for key in self.fields:
            length = max(length, self.get_field(key).shape[0])
        return length

    def insert(self, index: int, value: dict) -> None:
        raise NotImplementedError

    def __delitem__(self, key):
        raise NotImplementedError

    @property
    def shape(self):
        return (len(self),)

    def resize(self, *args):
        length = args[0]
        group = self._group
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
            shape = field.shape
            field.resize(length, *shape[1:])


@types.pytypes.register('numpy.ndarray', types=np.ndarray)
class NumpyArrayStrategy(types.PyType):
    def transform(self, data):
        return data

    def inverse_transform(self, data):
        if isinstance(data, np.ndarray):
            return data
        raise TypeError('expected \'np.ndarray\', found {}'.format(type(data).__name__))
