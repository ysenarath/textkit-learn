from __future__ import annotations

import functools

import nltk

__all__ = [
    "get_stopwords",
]


@functools.lru_cache(maxsize=5)
def get_stopwords(language: str = "english") -> frozenset[str]:
    """Return NLTK's stopwords for a language, downloading them if needed."""
    if language == "en":
        language = "english"
    try:
        words = nltk.corpus.stopwords.words(language)
    except LookupError:
        nltk.download("stopwords", quiet=True)
        words = nltk.corpus.stopwords.words(language)
    return frozenset(words)
