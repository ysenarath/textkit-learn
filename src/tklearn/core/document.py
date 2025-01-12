from __future__ import annotations

import abc
import weakref
from dataclasses import MISSING, dataclass, fields
from dataclasses import field as _field
from shlex import quote
from typing import Any, Generic, Mapping, Optional, TypeVar, get_origin
from typing import get_type_hints as _get_type_hints

import jmespath
import nltk
import torch
from datasets import Dataset, DatasetDict
from numpy import typing as nt
from typing_extensions import dataclass_transform

from tklearn.core.tasks import Future

T = TypeVar("T")

tweet_tokenizer = nltk.TweetTokenizer()


class DocumentWrapperForDataset(Mapping[str, Any]):
    def __init__(
        self,
        dataset: Dataset | DatasetDict,
        idx: int,
        split: Optional[str] = None,
    ):
        self._get_dataset = weakref.ref(dataset)
        self.idx = idx
        self.split = split

    @property
    def dataset(self) -> Dataset:
        if self.split is None:
            return self._get_dataset()
        return self._get_dataset()[self.split]

    @property
    def data(self) -> dict:
        return self.dataset[self.idx]

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.data})"


def mapped(**kwargs):
    if kwargs is None:
        kwargs = {}
    return _field(
        default=None,
        default_factory=MISSING,
        init=True,
        repr=True,
        hash=None,
        compare=True,
        metadata=kwargs,
    )


def get_type_hints(cls, globalns: Any = None, localns: Any = None):
    types = {}
    hints = _get_type_hints(cls, globalns=globalns, localns=localns)
    for field in fields(cls):
        if field.name not in hints:
            continue
        types[field.name] = hints[field.name]
    return types


class FieldType:
    def __getitem__(self, key: str) -> Any:
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key)

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class Mapped(FieldType, Generic[T]):
    def __new__(cls, doc: Any, field: str) -> str:
        value = jmespath.search(field, doc)
        return value


class Text(FieldType):
    def __init__(
        self,
        doc: Any,
        field: str,
        preprocessor: Any = None,
        encoder: Any = None,
        tokenizer: Any = None,
        vectorizer: Any = None,
        type: str = None,
    ):
        super().__init__()
        self.type = type
        self.raw: str = jmespath.search(field, doc)
        if self.raw is None:
            msg = f"field {field} not found in {doc}"
            raise ValueError(msg)
        self._cleaned: str | Future[str] = self.preprocess(preprocessor)
        self._tokens: Any | Future = self.tokenize(tokenizer)
        self._encoding: Any | Future = self.encode(encoder)
        self._embedding: (
            nt.NDArray | torch.Tensor | Future[nt.NDArray | torch.Tensor]
        ) = self.vectorize(vectorizer)

    @property
    def cleaned(self) -> str:
        if isinstance(self._cleaned, Future):
            self._cleaned = self._cleaned.result()
        return self._cleaned

    @property
    def tokens(self) -> Any:
        if isinstance(self._tokens, Future):
            self._tokens = self._tokens.result()
        return self._tokens

    @property
    def encoding(self) -> Any:
        if isinstance(self._encoding, Future):
            self._encoding = self._encoding.result()
        return self._encoding

    @property
    def embedding(self) -> Any:
        if isinstance(self._embedding, Future):
            self._embedding = self._embedding.result()
        return self._embedding

    def preprocess(self, preprocessor: Any = None) -> str:
        if preprocessor is None:
            return self.raw
        return preprocessor(self)

    def _default_tokenize(self) -> list[str]:
        return [
            tok
            for sent in nltk.sent_tokenize(self.cleaned)
            for tok in (
                tweet_tokenizer.tokenize(sent)
                if self.type and self.type.lower() == "tweet"
                else nltk.word_tokenize(sent)
            )
        ]

    def tokenize(self, tokenizer: Any = None) -> list[str] | Any:
        if tokenizer is None:
            return self._default_tokenize()
        return tokenizer(self)

    def encode(self, encoder: Any = None) -> Any:
        if encoder is None:
            return None
        return encoder(self)

    def vectorize(self, vectorizer: Any = None) -> Any:
        if vectorizer is None:
            return None
        return vectorizer(self)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(raw={quote(self.raw)})"

    def __str__(self) -> str:
        return self.raw


@dataclass_transform()
class DocumentMeta(abc.ABCMeta):
    def __new__(
        cls,
        __name: str,
        __bases: tuple[type, ...],
        __namespace: dict[str, Any],
        /,
        **kwargs,
    ):
        obj = super().__new__(cls, __name, __bases, __namespace)
        return dataclass(**kwargs)(obj)


class Document(Mapping, metaclass=DocumentMeta):
    data: dict[str, Any]

    def __post_init__(self):
        # init fields that are not initialized
        cls = self.__class__
        metadata = {f.name: f.metadata for f in fields(cls)}
        for name, type_ in get_type_hints(cls).items():
            # args = get_args(type_)
            type_ = get_origin(type_) or type_
            if not issubclass(type_, FieldType):
                continue
            value = type_(self, **metadata[name])
            setattr(self, name, value)

    def __getattr__(self, key: str) -> Any:
        if key in self.data:
            return self.data[key]
        return super().__getattr__(key)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __iter__(self):
        for field in fields(self):
            yield field.name

    def __len__(self) -> int:
        return len(fields(self))

    @classmethod
    def from_dataset(
        cls,
        dataset: Dataset | DatasetDict,
        idx: int,
        split: Optional[str] = None,
    ) -> "Document":
        return cls(DocumentWrapperForDataset(dataset, idx, split=split))
