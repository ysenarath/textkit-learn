"""Preprocessor class for text data."""

import re
import string
from dataclasses import dataclass, fields
from dataclasses import field as create_field
from typing import Any, List, Literal, Tuple, Union

from nltk.corpus import stopwords

from tklearn import lang, utils

OptionLabel = Literal["remove", "normalize", "ignore"]


class F:
    REMOVE = "remove"
    NORMALIZE = "normalize"
    IGNORE = "ignore"


def op_remove(pattern: re.Pattern, replacement: str, text: str):
    return pattern.sub("", text)


def op_normalize(pattern: re.Pattern, replacement: str, text: str):
    return pattern.sub(replacement, text)


def op_ignore(pattern: re.Pattern, replacement: str, text: str):
    return text


OPTION_LABEL_ACTIONS = {
    F.REMOVE: op_remove,
    F.NORMALIZE: op_normalize,
    F.IGNORE: op_ignore,
}


def escape(text: str):
    return text.replace("\\", "\\\\").replace("[", "\[").replace("]", "\]")


RE_URL = r"(?i)\b((?:[a-z][\w-]+:(?:\/{1,3}|[a-z0-9%])|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}\/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
RE_EMOTICON = r"(?:[<>]?[:;=8][\-o\*\']?[\)\]\(\[dDpP\/\:\}\{@\|\\]|[\)\]\(\[dDpP\/\:\}\{@\|\\][\-o\*\']?[:;=8][<>]?|<3)"
RE_NUMBER = r"(?:[+-]?[\d,]+(?:[.][\d,]+)?(?:[:\/][\d,]+(?:[.][\d,]+)?)?)"
RE_HTML_TAG = r"<[^>]+>"
RE_ASCII_ARROWS = r"[\-\=]+>|<[\-\=]+"
RE_USERNAME = r"(?:@[\w_]+)"
RE_HASHTAG = r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)"
RE_EMAIL = r"[\w.+-]+@[\w-]+\.(?:[\w-]\.?)+[\w-]"
RE_PUCTUATION = rf"[{re.escape(string.punctuation)}]"


@dataclass
class TextPreprocessor:
    urls: OptionLabel = create_field(
        default=F.NORMALIZE,
        metadata={
            "pattern": re.compile(RE_URL, re.VERBOSE | re.I | re.UNICODE),
            "replacement": "[URL]",
        },
    )
    emoticons: OptionLabel = create_field(
        default=F.IGNORE,
        metadata={
            "pattern": re.compile(RE_EMOTICON, re.VERBOSE | re.I | re.UNICODE),
            "replacement": "[EMOTICON]",
        },
    )
    html_tags: OptionLabel = create_field(
        default=F.REMOVE,
        metadata={
            "pattern": re.compile(RE_HTML_TAG, re.VERBOSE | re.I | re.UNICODE),
            "replacement": "[HTML_TAG]",
        },
    )
    ascii_arrows: OptionLabel = create_field(
        default=F.IGNORE,
        metadata={
            "pattern": re.compile(RE_ASCII_ARROWS, re.VERBOSE | re.I | re.UNICODE),
            "replacement": "[ASCII_ARROW]",
        },
    )
    usernames: OptionLabel = create_field(
        default=F.NORMALIZE,
        metadata={
            "pattern": re.compile(RE_USERNAME, re.VERBOSE | re.I | re.UNICODE),
            "replacement": "[USERNAME]",
        },
    )
    hashtags: OptionLabel = create_field(
        default=F.IGNORE,
        metadata={
            "pattern": re.compile(RE_HASHTAG, re.VERBOSE | re.I | re.UNICODE),
            "replacement": "[HASHTAG]",
        },
    )
    emails: OptionLabel = create_field(
        default=F.NORMALIZE,
        metadata={
            "pattern": re.compile(RE_EMAIL, re.VERBOSE | re.I | re.UNICODE),
            "replacement": "[EMAIL]",
        },
    )
    numbers: OptionLabel = create_field(
        default=F.IGNORE,
        metadata={
            "pattern": re.compile(RE_NUMBER, re.VERBOSE | re.I | re.UNICODE),
            "replacement": "[NUMBER]",
        },
    )
    punctuations: OptionLabel = create_field(
        default=F.IGNORE,
        metadata={
            "pattern": re.compile(RE_PUCTUATION, re.VERBOSE | re.I | re.UNICODE),
            "replacement": "[PUNCTUATION]",
        },
    )
    stopwords: Tuple[str] = create_field(
        default_factory=tuple,
        metadata={},
    )
    uncontract: bool = create_field(
        default=False,
        metadata={},
    )
    lowercase: bool = create_field(
        default=True,
        metadata={},
    )
    encoding: str = create_field(
        default="utf-8",
        metadata={},
    )

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "stopwords":
            if value is None:
                value = []
            elif isinstance(value, str):
                value = stopwords.words(value)
            value = tuple(sorted(set(value)))
            super().__setattr__(name, value)
        elif name == "lowercase":
            super().__setattr__(name, bool(value))
        elif name == "uncontract":
            super().__setattr__(name, bool(value))
        else:
            if value is None:
                value = "ignore"
            super().__setattr__(name, value)

    def remove_stopwords(self, text: str) -> str:
        if not hasattr(self, "_stopwords_pattern"):
            self._stopwords_pattern = {}
        if self.stopwords not in self._stopwords_pattern:
            self._stopwords_pattern.clear()
            base_pattern = "(?:" + "|".join(map(re.escape, self.stopwords)) + ")"
            self._stopwords_pattern[self.stopwords] = re.compile(
                rf"\s*\b{base_pattern}(?:'{base_pattern})?\b\s*(?=\s\b)",
                re.VERBOSE | re.I | re.UNICODE,
            )
        pattern: re.Pattern = self._stopwords_pattern[self.stopwords]
        return pattern.sub("", text)

    def preprocess(
        self, text: Union[List[Union[str, bytes]], Union[str, bytes]]
    ) -> List[str]:
        if isinstance(text, (str, bytes)):
            text = [text]
        return list(map(self._preprocess, text))

    def _preprocess(self, text: Union[str, bytes]) -> str:
        text_lang = lang.detect(text, default="en")
        x = escape(text if isinstance(text, str) else text.decode(self.encoding))
        if self.uncontract and text_lang == "en":
            x = lang.en.uncontract(x)
        if self.lowercase:
            x = str.lower(x)
        if self.stopwords:
            x = self.remove_stopwords(x)
        for field in fields(self):
            if "pattern" not in field.metadata:
                continue
            pattern = field.metadata["pattern"]
            if pattern is None:
                continue
            option_label = getattr(self, field.name)
            replacement = field.metadata["replacement"]
            action = OPTION_LABEL_ACTIONS[option_label]
            x = action(pattern, replacement, x)
        return x.strip()

    def _repr_html_(self) -> str:
        return utils.html.repr_html(self)
