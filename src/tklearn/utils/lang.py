from __future__ import annotations

import functools

import nltk


@functools.lru_cache(maxsize=1)
def get_stopwords(language: str = "english") -> set[str]:
    if language == "en":
        return get_stopwords("english")
    return set(nltk.corpus.stopwords.words(language))
