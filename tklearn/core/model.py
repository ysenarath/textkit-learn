from collections import UserDict
import typing

__all__ = [
    'Source',
    'Document',
]


class Source(object):
    def __init__(self, name, id=None):
        super().__init__()
        self.id = id
        self.name = name

    @classmethod
    def from_orm(cls, obj):
        return Source(
            id=obj.id,
            name=obj.name,
        )


class Document(UserDict):
    def __init__(
            self,
            data: dict[str, typing.Any],
            id: str = None,
            _id: int = None,
            source: typing.Union[Source, dict] = None,
    ):
        super().__init__(**data)
        self.source: Source = source
        self._id = _id
        self.id = str(id)

    @property
    def source(self):
        return self._source

    @source.setter
    def source(self, value):
        if isinstance(value, dict):
            value = Source(**value)
        elif value is not None and not isinstance(value, Source):
            raise TypeError(
                'source must be an instance of Source or a dict.'
            )
        self._source = value

    @classmethod
    def from_orm(cls, obj):
        source = None
        if hasattr(obj, 'source'):
            source = obj.source
        return Document(
            data=obj.data,
            id=obj.id,
            _id=obj._id,
            source=source,
        )
