from __future__ import annotations

import abc
import functools
from dataclasses import dataclass
from os import PathLike
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import nltk
import pandas as pd
from nltk.tokenize import TweetTokenizer as _TweetTokenizer
from nltk.tokenize import word_tokenize
from transformers import AutoTokenizer, PreTrainedTokenizer
from typing_extensions import ParamSpec

__all__ = [
    "Tokenizer",
    "FunctionTokenizer",
    "WordTokenizer",
    "TweetTokenizer",
    "WhitespaceTokenizer",
    "HuggingFaceTokenizer",
]

P = ParamSpec("P")
T = TypeVar("T")


class BatchTokenizeMethod(Generic[P, T]):
    def __init__(self, func: Callable[P, T], batched: bool = False):
        self.func = func
        self.batched = batched

    def __call__(
        self,
        tokenizer: Tokenizer,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> T:
        func = functools.partial(self.func, tokenizer)
        text, args = args[0], args[1:]
        if self.batched:
            if isinstance(text, str):
                text = [text]
            return func(text, *args, **kwargs)
        if isinstance(text, pd.Series):
            return text.apply(func, args=args, **kwargs)
        elif isinstance(text, list):
            return [func(t, *args, **kwargs) for t in text]
        elif isinstance(text, str):
            return [func(text, *args, **kwargs)]
        # error should point out the valid types
        msg = f"expected str, List[str], or pd.Series, but got {type(text).__name__}"
        raise TypeError(msg)

    def __get__(self, instance: Any, owner: Any) -> functools.partial:
        if instance is None:
            return self
        return functools.partial(self, instance)


class TokenizerMeta(abc.ABCMeta):
    def __new__(
        cls,
        name: str,
        bases: Tuple[Type, ...],
        namespace: Dict[str, Any],
        batched: bool = False,
        **kwargs,
    ) -> Type[Tokenizer]:
        if "tokenize" not in namespace:
            msg = "tokenize method is not implemented"
            raise TypeError(msg)
        tokenize = namespace["tokenize"]
        if not isinstance(tokenize, BatchTokenizeMethod):
            tokenize = BatchTokenizeMethod(tokenize, batched=batched)
        namespace["tokenize"] = tokenize
        cls = super().__new__(cls, name, bases, namespace, **kwargs)
        return cls


class Tokenizer(Generic[P, T], metaclass=TokenizerMeta):
    @abc.abstractmethod
    def tokenize(self, *args: P.args, **kwargs: P.kwargs) -> T:
        raise NotImplementedError

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T:
        return self.tokenize(*args, **kwargs)


class FunctionTokenizer(Tokenizer[P, T]):
    def __init__(self, func: Callable[P, T]):
        self._func = func

    def tokenize(self, *args: P.args, **kwargs: P.kwargs) -> T:
        return self._func(*args, **kwargs)


class WhitespaceTokenizer(Tokenizer):
    def tokenize(self, text: str) -> List[str]:
        return text.split()


@dataclass
class WordTokenizer(Tokenizer):
    language: str = "en"
    preserve_line: bool = False

    def __post_init__(self):
        if self.language.lower() == "en":
            self.language = "english"
        nltk.download("punkt")

    def tokenize(self, text: str) -> List[str]:
        return word_tokenize(text, language=self.language)


@dataclass
class TweetTokenizer(Tokenizer):
    preserve_case: bool = True
    reduce_len: bool = False
    strip_handles: bool = False
    match_phone_numbers: bool = True

    def __post_init__(self):
        self._tweet_tokenizer = _TweetTokenizer(
            preserve_case=self.preserve_case,
            reduce_len=self.reduce_len,
            strip_handles=self.strip_handles,
            match_phone_numbers=self.match_phone_numbers,
        )

    def tokenize(self, text: str) -> List[str]:
        return self._tweet_tokenizer.tokenize(text)


class HuggingFaceTokenizer(Tokenizer, Generic[P, T], batched=True):
    def __init__(self, tokenizer: Callable[P, T]):
        self.hf_tokenizer = tokenizer

    def tokenize(self, *args: P.args, **kwargs: P.kwargs) -> T:
        return self.hf_tokenizer(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        if hasattr(self.hf_tokenizer, name):
            return getattr(self.hf_tokenizer, name)
        raise AttributeError

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, PathLike[str]],
        *inputs: Any,
        **kwargs: Any,
    ):
        tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path, *inputs, **kwargs
        )
        return HuggingFaceTokenizer(tokenizer)
