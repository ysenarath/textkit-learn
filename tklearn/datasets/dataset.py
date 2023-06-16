"""Dataset module.

A module for working with datasets.
"""
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
    """A mapping of fields to arrays."""

    def __init__(self, data, schema=None):
        """Initialize a new FieldMapping.

        Parameters
        ----------
        data : MutableMapping
            The data of the field mapping.
        schema : Schema
            The schema of the field mapping.
        """
        self._data = data
        self._schema = schema

    def __getitem__(self, key):
        """Return the field at the given key.

        Parameters
        ----------
        key : str
            The key of the field to return.

        Returns
        -------
        field : array-like
            The field at the given key.

        Raises
        ------
        KeyError
            If the key does not exist.
        """
        data, schema = self._data[key], self._schema[key]
        return DocumentList(fields=FieldMapping(data, schema))

    def __setitem__(self, key, value):
        """Set the value of the field at the given key.

        Parameters
        ----------
        key : str
            The key of the field to set.
        value : array-like
            The value of the field to set.

        Raises
        ------
        ValidationError
            If the value is not compatible with the current schema.
        NotImplementedError
            If the value is not an array-like.
        """
        if isinstance(value, np.ndarray):
            arr = self._data.create_dataset(
                key, shape=value.shape,
                fill_value=None, dtype=value.dtype
            )
            arr[:] = value
            self._schema = Schema.from_data(value)
        raise NotImplementedError

    def __delitem__(self, key):
        """Delete the field at the given key. (Not implemented)

        Parameters
        ----------
        key : str   
            The key of the field to delete.

        Raises
        ------
        KeyError
            If the key does not exist.
        """
        raise NotImplementedError

    def __len__(self):
        """Return the length of the array.

        Returns
        -------
        length: int
            The length of the array.
        """
        if isinstance(self._data, zarr.Group):
            return len(self._data)
        return 0

    def __iter__(self):
        """Return an iterator over the keys of the array.

        Returns
        -------
        keys : Iterator[str]
            An iterator over the keys of the array.
        """
        keys = []
        if isinstance(self._data, zarr.Group):
            self._data.visitkeys(
                lambda x: keys.append(x) if isinstance(
                    self._data[x], zarr.core.Array) else None
            )
        for key in keys:
            yield key

    def resize(self, *args):
        """Resize the array.

        Parameters
        ----------
        *args : int
            The new shape of the array.
        """
        # todo: check if following error is raised
        # Raises
        # ------
        # ValueError
        #     If the new shape is not compatible with the current shape.
        if isinstance(self._data, zarr.Array):
            self._data.resize(*args)
        else:
            length = args[0]
            group = self._data
            _, schema = self._schema.normalize(
                None, return_schema=True, validate=False)
            for key, prop in schema.properties.items():
                if key in group:
                    continue
                if hasattr(prop.items.logical_type, 'create_dataset'):
                    prop.items.logical_type.create_dataset(
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
            for key, prop in schema.properties.items():
                field = self.fields[key]
                if not isinstance(field._data, zarr.Array):
                    continue
                shape = field._data.shape
                field.resize(length, *shape[1:])


class DocumentList(Sequence):
    """A list of documents."""

    IndexerType = typing.Union[int, slice, tuple[typing.Union[int, slice]]]

    def __init__(self, fields: FieldMapping):
        """Initialize a new DocumentList.

        Parameters
        ----------
        fields : FieldMapping
            The fields of the document list.
        """
        self.fields = fields

    @property
    def _data(self):
        """Return the data of the array."""
        return self.fields._data

    @property
    def _schema(self):
        """Return the schema of the array."""
        return self.fields._schema

    def __getitem__(self, index):
        """Return the document at the given index.

        Parameters
        ----------
        index : int, slice, tuple[int, ...], tuple[slice, ...]
            The index of the document to return.

        Returns
        -------
        doc : dict
            The document at the given index.

        Raises
        ------
        TypeError
            If the index is not an integer, a sequence of integers or a slice.
        ValueError
            If the index is a sequence of integers and the length of the sequence 
            is not equal to the length of the array.
        """
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
        """Set the value of the document at the given index.


        Parameters
        ----------
        index : int, slice, tuple[int, ...], tuple[slice, ...]
            The index of the document to set.
        value : dict
            The value of the document to set.

        Raises
        ------
        ValidationError
            If the value is not compatible with the current schema.
        """
        self._schema.items.validate(value)
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
            doc, schema = self._schema.items.normalize(
                value, return_schema=True)
            for key, prop in schema.properties.items():
                value = doc.get(key)
                prop = schema.properties[key]
                if value is None and prop.type not in {'object', 'array'}:
                    value = zarr.empty(1, dtype=prop.type)[0]
                field = self.fields[key]
                field[index] = value

    def append(self, doc: dict, dynamic=True):
        """Append a document to the array.

        Parameters
        ----------
        doc : dict
            The document to append.
        dynamic : bool  (default=True)  
            If True, the schema will be updated with the new document.

        Raises
        ------
        ValidationError
            If the document is not compatible with the current schema.
        """
        if dynamic:
            schema = Schema.from_data(doc)
            self._schema.items.update(schema)
        self.resize(len(self) + 1)
        self[-1] = doc

    def __len__(self):
        """Return the length of the array.

        Returns
        -------
        length: int
            The length of the array.
        """
        if isinstance(self._data, zarr.Array):
            return self._data.shape[0]
        length = 0
        for key in self.fields:
            length = max(length, self.fields[key].shape[0])
        return length

    @property
    def shape(self):
        """Return the shape of the array.

        Returns
        -------
        shape: tuple[int]
            The shape of the array.
        """
        return (len(self),)

    def _index_to_list(self, index):
        """Convert index to list of indices.

        Parameters
        ----------
        index: int, slice, tuple[int, ...], tuple[slice, ...]
            The index to convert.

        Returns
        -------
        index_iter: list[int]
            The list of indices.
        """
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


@types.logical_types.register('numpy.ndarray', types=np.ndarray)
class NumpyArrayType(types.LogicalType):
    """A logical type for numpy arrays."""

    def encode(self, data):
        """Encode a numpy array.

        Parameters
        ----------
        data: np.ndarray
            The numpy array to encode.

        Returns
        -------
        data: np.ndarray
            The encoded numpy array.
        """
        if isinstance(data, np.ndarray):
            return data
        return np.array(data)

    def decode(self, data):
        """Decode a numpy array.

        Parameters
        ----------
        data: np.ndarray
            The numpy array to decode.

        Returns
        -------
        data: np.ndarray
            The decoded numpy array.

        Raises
        ------
        TypeError
        If the data is not a numpy array.
        """
        if isinstance(data, np.ndarray):
            return data
        raise TypeError(
            'expected \'np.ndarray\', found {}'.format(type(data).__name__))

    @staticmethod
    def create_dataset(group, key, schema):
        """Create a dataset for a numpy array.

        Parameters
        ----------
        group: zarr.Group
            The group to create the dataset in.
        key: str
            The key to create the dataset with.
        schema: Schema
            The schema of the dataset.

        Returns
        -------
        dataset: zarr.Dataset
            The created dataset.
        """
        group.create_dataset(
            key, shape=(0, *schema.shape),
            dtype=schema.dtype
        )


class Dataset(ObserverMixin, DocumentList):
    """A dataset is a collection of documents with a schema."""

    @typing.overload
    def __init__(self, path: typing.Optional[typing.Union[zarr.storage.BaseStore, MutableMapping, str]] = None):
        """Create a new dataset.

        Parameters
        ----------
        path: str
            The path to the zarr file to load.
        """
        ...

    @typing.overload
    def __init__(self, data: typing.Optional[typing.Union[zarr.Group, zarr.Array]] = None):
        ...

    def __init__(self, *args, **kwargs):
        """Create a new dataset.

        Examples
        --------
        >>> import zarr
        >>> from tklearn.datasets import Dataset
        >>> dataset = Dataset()
        >>> dataset.append({'a': 1, 'b': 2})

        Parameters
        ----------
        data: zarr.Group or zarr.Array
            The data to load into the dataset. Can be a path to a zarr file, a zarr group or array, or a zarr store.
        """
        data = kwargs.get('data', kwargs.get(
            'path', args[0] if len(args) > 0 else None
        ))
        if isinstance(data, str):
            data = zarr.open(data, mode='a')
        if data is None:
            data = zarr.group()
        elif not isinstance(data, (zarr.Group, zarr.Array)):
            raise TypeError('expected \'zarr.Group\' or \'zarr.Array\', found {}'.format(
                type(data).__name__))
        schema = None
        # load schema from data
        if data is not None and 'schema' in data.attrs:
            schema_dict = data.attrs['schema']  # type: dict
            schema = Schema.from_dict(schema_dict)
        if schema is None:
            schema = Schema('array', items=Schema())
        super(Dataset, self).__init__(fields=FieldMapping(data, schema))
        # for root group only attach it to sync the schema
        self._schema.observers.attach(self)
        self.notify()

    @property
    def metadata(self):
        """The metadata of the dataset.

        Returns
        -------
        metadata: dict
            The metadata of the dataset.
        """
        # "metadata" is a special attribute that is stored in the root group of the zarr hierarchy
        # return metadata
        return self._data.attrs['metadata']

    def notify(self, *args, **kwargs):
        """Notify the observers of a change in the schema.

        Parameters
        ----------
        *args:
            The arguments to pass to the observers.
        **kwargs:
            The keyword arguments to pass to the observers.

        Returns
        -------
        None
        """
        # update dtypes of data or data shapes according to changed schema schema
        _, schema = self._schema.normalize(None,return_schema=True,validate=False)
        for key in self.fields.keys():
            field = self.fields[key]
            if field._data.dtype != schema[key].dtype:
                field.dtype = schema[key].dtype
        self._data.attrs['schema'] = self._schema.to_dict()
        # revert schema to original schema
        schema_dict = self._data.attrs['schema']  # type: dict
        self._schema = Schema.from_dict(schema_dict)
