from collections.abc import Sequence
import typing

from pydantic import BaseModel

from tklearn.datasets.document import Document

DocumentCollection = typing.List[typing.Union['Document', 'Collection']]

__all__ = ['Collection']


def for_each(items: DocumentCollection, attr: str) -> typing.Any:
    """Get attribute of each item in a list."""
    item_attrs = []
    is_collection = True
    for item in items:
        output = getattr(item, attr)
        if not isinstance(output, (Document, Collection)):
            is_collection = False
        item_attrs.append(output)
    if is_collection:
        return Collection(items=item_attrs)
    return item_attrs


class Collection(BaseModel, Sequence):
    """Collection of Documents or other Collections."""
    items: DocumentCollection
    offset: typing.Optional[int] = None

    def __getattr__(self, attr) -> typing.Any:
        return for_each(self.items, attr)

    def __getitem__(self, index) -> typing.Any:
        if self.offset is None:
            return self.items[index]
        return self.items[index - self.offset]

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self) -> typing.Iterator:
        return iter(self.items)

    def __add__(self, other: 'Collection') -> 'Collection':
        return Collection(items=self.items + other.items, offset=self.offset)
