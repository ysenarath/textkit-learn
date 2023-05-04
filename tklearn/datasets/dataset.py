import typing
from collections.abc import Iterable, Sequence, MutableMapping

import numcodecs
import numpy as np
import zarr

from tklearn.datasets import types
from tklearn.datasets.schema import Schema
from tklearn.utils.observable import ObserverMixin

__all__ = [
    'DocumentList',
    'Dataset',
]


class FieldMapping(MutableMapping):
    def __init__(self, docs: 'DocumentList'):
        self._docs = docs  # type: DocumentList

    def __getitem__(self, key):
        return DocumentList(self._docs._data[key], schema=self._docs._schema[key])

    def __setitem__(self, key, value):
        if isinstance(value, np.ndarray):
            arr = self._docs._data.create_dataset(
                key, shape=value.shape,
                fill_value=None, dtype=value.dtype
            )
            arr[:] = value
            self._docs._schema = Schema.from_data(value)
        raise NotImplementedError

    def __delitem__(self, key):
        raise NotImplementedError

    def __len__(self):
        if isinstance(self._docs._data, zarr.Group):
            return len(self._docs._data)
        return 0

    def __iter__(self):
        keys = []
        if isinstance(self._docs._data, zarr.Group):
            self._docs._data.visitkeys(
                lambda x: keys.append(x) if isinstance(self._docs._data[x], zarr.core.Array) else None
            )
        for key in keys:
            yield key


class DocumentList(Sequence):
    IndexerType = typing.Union[int, slice, tuple[typing.Union[int, slice]]]

    def __init__(self, data: typing.Union[zarr.Group, zarr.Array] = None, schema: typing.Optional[Schema] = None):
        if data is None:
            data = zarr.group()
        self._data = data
        if schema is None:
            schema = Schema('array', items=Schema())
        self._schema = schema
        if isinstance(self._data, zarr.Group):
            self.fields = FieldMapping(self)
        else:
            self.fields = None

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
            for key, value in self.fields.items():
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
                field = self.fields[key]
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
            length = max(length, self.fields[key].shape[0])
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
                    if prop.items.pytype is not None and hasattr(prop.items.pytype, 'create_dataset'):
                        prop.items.pytype.create_dataset(
                            group, key, prop.items
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
                field = self.fields[key]
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
class NumpyArrayType(types.PyType):
    def encode(self, data):
        return data

    def decode(self, data):
        if isinstance(data, np.ndarray):
            return data
        raise TypeError('expected \'np.ndarray\', found {}'.format(type(data).__name__))

    @staticmethod
    def create_dataset(group, key, schema):
        group.create_dataset(
            key, shape=(0, *schema.shape),
            dtype=schema.dtype
        )


class Dataset(ObserverMixin, DocumentList):
    """
    A more or less complete user-defined wrapper around DocumentArray objects.
    """

    @typing.overload
    def __init__(self, path: typing.Optional[typing.Union[zarr.storage.BaseStore, MutableMapping, str]]):
        ...

    def __init__(self, data: typing.Optional[typing.Union[zarr.Group, zarr.Array]] = None):
        if isinstance(data, str):
            data = zarr.open(data, mode='r+')
        schema = None
        if data is not None:
            schema_dict = self._data.attrs['schema']  # type: dict
            schema = Schema.from_dict(schema_dict)
        super(Dataset, self).__init__(data, schema)
        # for root group only attach it to sync the schema
        self._schema.observers.attach(self)
        self.notify()

    def notify(self, *args, **kwargs):
        # schema has been updated
        self._data.attrs['schema'] = self._schema.to_dict()

    @property
    def metadata(self):
        return self._data.attrs['metadata']
