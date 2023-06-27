"""DataStore class and its subclasses.

DataStore is a class that defines the interface for storing and retrieving
data. It is used by the Dataset class to store and retrieve data.

The SQLAlchemyDataStore class is a subclass of DataStore that uses SQLAlchemy
to store and retrieve data. It is used by the Dataset class to store and
retrieve data.
"""
import abc
from functools import wraps
import typing
from tklearn.core.env import Environment
from tklearn.core.model import Document, Source
from tklearn.datasets.source import DataSource


from tklearn.utils import logging

__all__ = [
    'DataStore',
]

logger = logging.get_logger(__name__)


registery = {}


class DataStore(object, metaclass=abc.ABCMeta):
    def __new__(cls, name, args=None, type=None) -> 'DataStore':
        if type is not None and type in registery and cls is DataStore:
            return registery[type](name=name, args=args)
        self = super(DataStore, cls).__new__(cls)
        self._name: str = name
        self.args: typing.Dict[str, typing.Any] = {}
        self.type: str = type
        self.__env__ = None
        return self

    @property
    def name(self) -> str:
        return self._name

    @property
    def env(self) -> Environment:
        return self.__env__

    @abc.abstractmethod
    def add(self, doc: typing.Union[Document, typing.Iterable[Document]]):
        raise NotImplementedError

    @abc.abstractmethod
    def delete(self, source: Source):
        raise NotImplementedError

    @abc.abstractmethod
    def get(self, id: int):
        raise NotImplementedError

    @abc.abstractmethod
    def count(self):
        raise NotImplementedError

    @abc.abstractmethod
    def __iter__(self) -> iter:
        raise NotImplementedError

    def all(self) -> list[Document]:
        return list(self)

    def _get_document_id(self, doc: typing.Union[dict, Document], key=None) -> str:
        if key is None:
            key = self.args.get('id_field_name', 'id')
        if isinstance(key, str):
            key = [key]
        for k in key:
            doc = doc[k]
        return str(doc)

    def write(self, source: typing.Union[DataSource, 'DataStore'],
              verbose: bool = False) -> bool:
        base_source = Source(name=source.name)
        if hasattr(source, 'name'):
            self.delete(source=base_source)
        if isinstance(source, DataSource):
            chunk = (Document(doc, id=self._get_document_id(doc),
                     source=base_source) for doc in source.read())
        elif isinstance(source, DataStore):
            chunk = (doc for doc in source)
        if verbose:
            desc = f'Writing data from {source.name} to {self.name}'
            chunk = self.env.progress(chunk, total=len(source), desc=desc)
        self.add(chunk)
        return True

    def __len__(self) -> int:
        return self.count()

    def dict(self) -> dict:
        return {
            'name': self.name,
            'args': self.args,
            'type': self.type,
        }


def register(_type: str):
    def _register(cls: type):
        @wraps(cls)
        def get_instance(name, args=None):
            # automatically innject type attr of the object
            return cls(name=name, args=args, type=_type)
        registery[_type] = get_instance
        return get_instance
    return _register
