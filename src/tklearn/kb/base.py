from __future__ import annotations

import functools
from collections import defaultdict
from collections.abc import Iterable
from typing import Any, ClassVar

import nltk
import numpy as np
from nightjar import AutoModule, BaseConfig, BaseModule
from nltk.corpus import stopwords

from tklearn.kb.lexicon import Lexicon
from tklearn.kb.models import Candidate, Mention, Span, Triple
from tklearn.kb.triple_store import TripleStore

nltk.download("stopwords", quiet=True)


@functools.lru_cache(maxsize=1)
def get_stopwords(language: str = "english") -> set[str]:
    return set(stopwords.words(language))


class ArtifactStoreConfig(BaseConfig, dispatch="name"):
    name: ClassVar[ArtifactStore]


class ArtifactStore(BaseModule):
    config: ArtifactStoreConfig
    # form (str) -> set of words (same form may have different words)
    lexicon: Lexicon[set[str]]
    triples: TripleStore
    gloss2idx: dict[str, int]
    idx2gloss: dict[int, str]
    # word (str) -> set of senses (int)
    senses: dict[str, set[int]]
    embeddings: dict[int, np.ndarray]
    attrs: dict[str, set[int]]


class AutoArtifactStore(AutoModule):
    def __new__(cls, config: Any) -> ArtifactStore:
        if isinstance(config, ArtifactStore):
            return config
        if isinstance(config, str):
            config = {"name": config}
        if not isinstance(config, ArtifactStoreConfig):
            config = ArtifactStoreConfig.from_dict(config)
        return super().__new__(cls, config)


class KnowledgeBase:
    lexicon: Lexicon[set[str]]
    triples: TripleStore
    gloss2idx: dict[str, int]
    idx2gloss: dict[int, str]
    senses: dict[str, set[int]]
    embeddings: dict[int, np.ndarray]
    attrs: dict[str, set[int]]

    def __init__(self, config: Any):
        self.store = AutoArtifactStore(config)
        index_subject = {}
        for triple in self.triples.query():
            if triple.subject not in index_subject:
                index_subject[triple.subject] = {}
            if triple.predicate not in index_subject[triple.subject]:
                index_subject[triple.subject][triple.predicate] = set()
            index_subject[triple.subject][triple.predicate].add(triple.object)
        self.index_subject = index_subject

    def __getattr__(self, name: str) -> Any:
        try:
            return getattr(self.store, name)
        except AttributeError:
            return super().__getattr__(name)

    def extract_candidates(self, word: str) -> Iterable[Candidate]:
        """Get all words and senses for a given form."""
        # get all the senses of the word
        for sense_id in self.senses.get(word, None) or []:
            gloss = self.idx2gloss[sense_id]
            embedding = self.embeddings[sense_id]
            yield Candidate(word, gloss, embedding, sense_id).bind(self)

    def extract_mentions(
        self,
        text: str,
        min_word_len: int = 3,
        stopwords: Iterable[str] | str = "english",
    ) -> Iterable[Mention]:
        """Analyze text and return a list of words and their senses."""
        stopwords = get_stopwords(stopwords)
        stopwords = set(stopwords)
        for words, start, end in self.lexicon.extract(text):
            candidates = []
            form = text[start:end]
            if len(form) < min_word_len:
                continue
            if form.lower() in stopwords:
                continue
            for word in words:
                for cc in self.extract_candidates(word):
                    if cc.word.lower() != form.lower():
                        continue
                    candidates.append(cc)
            yield Mention(form, Span(start, end), candidates)

    def augment(
        self, text: str, mentions=None, filter_func=None
    ) -> Iterable[dict[str, Any]]:
        """Augment text by replacing words with synonyms, hyponyms, and hypernyms."""
        filter_func = filter_func or (lambda x: True)
        yield {
            "text": text,
            "original": text,
            "span.start": 0,
            "span.end": 0,
            "relations": [],
            "support": 0,
        }
        if mentions is None:
            mentions = self.extract_mentions(text)
        for mention in mentions:
            start, end = mention.span
            prefix, suffix = text[:start], text[end:]
            aug_words = defaultdict(set)
            for candidate in mention.candidates:
                subject = (candidate.word, candidate.sense_id)
                for predicate in ["synonym", "hyponym", "instance"]:
                    objects = self.index_subject.get(subject, {}).get(
                        predicate, []
                    )
                    for object_ in objects:
                        rel = Triple(subject, predicate, object_)
                        if not filter_func(rel):
                            continue
                        aug_words[rel.object[0]].add(rel.to_tuple())
            for word, relations in aug_words.items():
                augmented_text = None
                augmented_text = prefix + word + suffix
                yield {
                    "text": augmented_text,
                    "original": text,
                    "span.start": start,
                    "span.end": end,
                    "relations": list(relations),
                    "support": len(relations),
                }
