from __future__ import annotations

import weakref
from collections import deque
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Generic,
    Mapping,
    MutableMapping,
    Optional,
    Protocol,
    TypeVar,
    runtime_checkable,
)

import nltk
from datasets import Dataset, DatasetDict
from datasets.fingerprint import Hasher
from nltk import TweetTokenizer
from typing_extensions import Concatenate, ParamSpec

__all__ = [
    "Document",
]

P = ParamSpec("P")
T = TypeVar("T")

tweet_tokenizer = TweetTokenizer()


@runtime_checkable
class Future(Protocol):
    def result(self) -> Any:
        raise NotImplementedError


class DocumentWrapper(Mapping[str, Any]):
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


class FieldType:
    def __getitem__(self, key: str) -> Any:
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key)

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class Text(FieldType):
    def __init__(
        self,
        data: str,
        column: str = "text",
        preprocessor: Callable[[Text], str] | None = None,
        tokenizer: Any = None,
        vectorizer: Any = None,
        type: str = None,
    ) -> None:
        self.type = type
        self.raw = data[column]
        self.fingerprint = Hasher.hash((
            self.raw,
            preprocessor,
            tokenizer,
            vectorizer,
            self.type,
        ))
        if preprocessor:
            self._cleaned = preprocessor(self)
        else:
            self._cleaned = self.raw
        if tokenizer:
            self._tokens = tokenizer(self)
        else:
            self._tokens = self._default_tokenizer()
        if vectorizer:
            self._embedding = vectorizer(self)
        else:
            self._embedding = None

    @property
    def cleaned(self) -> str:
        if isinstance(self._cleaned, Future):
            self._cleaned = self._cleaned.result()
        return self._cleaned

    @property
    def tokens(self) -> list[str]:
        if isinstance(self._tokens, Future):
            self._tokens = self._tokens.result()
        return self._tokens

    @property
    def embedding(self) -> Any:
        if isinstance(self._embedding, Future):
            self._embedding = self._embedding.result()
        return self._embedding

    def _default_tokenizer(self) -> list[str]:
        return [
            tok
            for sent in nltk.sent_tokenize(self.cleaned)
            for tok in (
                tweet_tokenizer.tokenize(sent)
                if self.type.lower() == "tweet"
                else nltk.word_tokenize(sent)
            )
        ]

    def __repr__(self):
        qtext = self.raw
        if len(qtext) > 20:
            qtext = qtext[:17] + "..."
        qtext = qtext.replace('"', '\\"')
        return f'{self.__class__.__name__}("{qtext}")'

    def __str__(self):
        return self.raw


class Field(Generic[P, T]):
    def __init__(
        self,
        __func: Callable[Concatenate[Any, P], T],
        /,
        *args: P.args,
        **kwargs: P.kwargs,
    ):
        self.__func = __func
        self.args = args
        self.kwargs = kwargs

    def __call__(self, data: Mapping[str, Any]) -> T:
        return self.__func(data, *self.args, **self.kwargs)


class Document(MutableMapping[str, Any]):
    def __init__(
        self,
        data: Mapping[str, Any],
        mapping: Dict[str, Any] | None = None,
    ) -> None:
        self.data = data
        if mapping is None and "text" in self.data:
            mapping = {"text": Field(Text, "text")}
        elif mapping is None:
            mapping = {}
        self.mapping = mapping
        self.cache = {}
        deque(self[key] for key in mapping)

    @property
    def dataset(self) -> Optional[Dataset]:
        return getattr(self.data, "dataset", None)

    def __setitem__(self, key: str, value: Any) -> None:
        self.data[key] = value
        # invalidate cache
        self.cache = {}

    def __delitem__(self, key: str) -> None:
        del self.data[key]
        # invalidate cache
        self.cache = {}

    def __getitem__(self, key: str) -> Any:
        if key in self.mapping:
            if key not in self.cache:
                func = self.mapping[key]
                self.cache[key] = func(self.data)
            return self.cache[key]
        return self.data[key]

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError:
            cls = self.__class__.__name__
            msg = f"'{cls}' object has no attribute '{key}'"
            raise AttributeError(msg)

    def __iter__(self) -> Generator[str, None, None]:
        return iter(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.data})"

    @classmethod
    def from_dataset(
        cls,
        dataset: Dataset | DatasetDict,
        idx: int,
        split: Optional[str] = None,
    ) -> "Document":
        return cls(DocumentWrapper(dataset, idx, split=split))
