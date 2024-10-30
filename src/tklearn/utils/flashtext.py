from __future__ import annotations

import pickle
from pathlib import Path

from flashtext2 import KeywordProcessor as BaseKeywordProcessor

__all__ = [
    "KeywordProcessor",
]


class KeywordProcessor:
    def __init__(self, case_sensitive: bool = False):
        self._processor = BaseKeywordProcessor(case_sensitive=case_sensitive)

    def extract_keywords(self, text: str, span_info: bool = False):
        if span_info:
            return self._processor.extract_keywords_with_span(text)
        return self._processor.extract_keywords(text)

    def replace_keywords(self, text: str):
        return self._processor.replace_keywords(text)

    def dump(self, path: Path | str) -> None:
        path = Path(path)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path | str) -> KeywordProcessor:
        path = Path(path)
        with open(path, "rb") as f:
            return pickle.load(f)
