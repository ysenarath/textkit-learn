from __future__ import annotations

import functools
import re

import nltk


@functools.lru_cache(maxsize=1)
def get_stopwords(language: str = "english") -> set[str]:
    if language == "en":
        return get_stopwords("english")
    return set(nltk.corpus.stopwords.words(language))


class Escaper:
    def __init__(self, chars: str, escape_char: str = "\\"):
        self._escape_char = escape_char
        if escape_char not in chars:
            chars += escape_char
        self._translation_table = {}
        for i in chars:
            self._translation_table.update(str.maketrans({i: escape_char + i}))
        self._unescape_pattern = re.compile(rf"{re.escape(escape_char)}(.)")

    @property
    def escape_char(self) -> str:
        return self._escape_char

    def escape(self, text: str | bytes) -> str | bytes:
        """Escape special characters in a string."""
        if isinstance(text, str):
            return text.translate(self._translation_table)
        else:
            text = str(text, "latin1")
            return text.translate(self._translation_table).encode("latin1")

    def unescape(self, text: str | bytes) -> str | bytes:
        """Unescape special characters in a string."""
        if isinstance(text, str):
            return self._unescape_pattern.sub(r"\1", text)
        else:
            text = str(text, "latin1")
            return self._unescape_pattern.sub(r"\1", text).encode("latin1")


class Codec:
    def __init__(self, delimiter: str = "\t"):
        self.delimiter = delimiter
        self.escaper = Escaper(delimiter)
        ec = re.escape(self.escaper.escape_char)
        self.split_pattern = re.compile(
            rf"(?<!{ec})(?:{ec}{ec})*{re.escape(delimiter)}"
        )

    def encode(self, obj: tuple[str]) -> bytes:
        return self.delimiter.join(
            self.escaper.escape(arg) for arg in obj
        ).encode()

    def decode(self, data: bytes) -> tuple[str]:
        if not isinstance(data, str):
            data = data.decode()
        parts = self.split_pattern.split(data)
        return tuple(self.escaper.unescape(arg) for arg in parts)


codec = Codec()
