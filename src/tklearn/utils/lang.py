from __future__ import annotations

import functools

import nltk

__all__ = [
    "get_stopwords",
]


@functools.lru_cache(maxsize=5)
def get_stopwords(language: str = "english") -> set[str]:
    if language == "en":
        return get_stopwords("english")
    return set(nltk.corpus.stopwords.words(language))
