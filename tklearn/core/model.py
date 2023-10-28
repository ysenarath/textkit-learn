from __future__ import annotations
from collections.abc import Mapping, Sequence
from typing import Iterator, List, Optional, Union, NamedTuple

from sortedcontainers import SortedKeyList


class Span(NamedTuple):
    start: int
    end: int

    def to_slice(self) -> slice:
        return slice(
            self.start,
            self.end,
        )

    def __repr__(self) -> str:
        return f"<Span ({self.start}, {self.end})>"


class Annotation(object):
    def __init__(
        self,
        doc: TextDocument,
        span: Optional[Span] = None,
        label: str = None,
    ):
        self.doc = doc
        self.span = span
        self.label = label

    @property
    def text(self) -> str:
        return self.doc.text[self.span.to_slice()]

    @classmethod
    def isinstance(cls, obj) -> bool:
        return isinstance(obj, cls)

    def get_order(self) -> int:
        return self.span.start if self.span else 0


class AnnotationList(SortedKeyList):
    def __init__(self, iterable=None):
        super().__init__(iterable, key=Annotation.get_order)


class Token(Annotation):
    def __init__(self, doc: Document, span: Span) -> None:
        super(Token, self).__init__(doc, span)


class Document(Mapping):
    def __init__(self, *args, **kwargs) -> None:
        super(Document, self).__init__()
        self.attrs = dict(*args, **kwargs)

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as _:
            raise AttributeError(name)

    def __getitem__(self, key):
        return self.attrs[key]

    def __len__(self) -> int:
        return len(self.attrs)

    def __iter__(self) -> Iterator:
        yield from self.attrs


class TextDocument(Document):
    def __init__(self, *args, **kwargs) -> None:
        data = dict(*args, **kwargs)
        super(TextDocument, self).__init__(data)
        self.annotations: AnnotationList = AnnotationList()
        self.embedding = None

    @property
    def text(self):
        return self["text"]

    @property
    def tokens(self) -> List[Token]:
        return list(filter(Token.isinstance, annotations))

    def __repr__(self):
        return f"{self.text}"


class MappingFunction:
    def __init__(self, func: callable) -> None:
        self.func = func

    def __call__(self, docs: Union[Mapping, Sequence]):
        if isinstance(docs, Mapping) or isinstance(docs, str):
            return self.func(docs)
        return list(map(self.func, docs))


def mapping(func: callable):
    return MappingFunction(func)
