from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from typing import Any, ClassVar

import numpy as np
from nightjar import AutoModule, BaseConfig, BaseModule
from typing_extensions import Protocol

from tklearn.kb.lexicon import Lexicon
from tklearn.kb.models import Candidate, Mention, Span, Triple
from tklearn.utils.lang import get_stopwords


class ArtifactStoreConfig(BaseConfig, dispatch="name"):
    name: ClassVar[str]


class TripleStore(Protocol):
    # def get(self, subj: S, default: SV) -> dict[P, O]: ...
    def get(
        self, subj: str | tuple[str, int], default: Any = None
    ) -> dict[str, list[tuple[str, int]]]: ...


class ArtifactStore(BaseModule):
    config: ArtifactStoreConfig

    # form (str) -> set of words (set[str])
    lexicon: Lexicon[set[str]]
    # subject (str, int) -> {predicate (str) -> [object (str, int)]}
    triples: TripleStore
    # gloss (str) -> index (int)
    gloss2idx: dict[str, int]
    # index (int) -> gloss (str)
    idx2gloss: dict[int, str]
    # word (str) -> set of senses (set[int])
    senses: dict[str, set[int]]
    # sense (int) -> embedding (np.ndarray)
    embeddings: dict[int, np.ndarray]
    # attribute (str) -> set of senses (set[int])
    attrs: dict[str, set[int]]


class AutoArtifactStore(AutoModule):
    def __new__(cls, config: Any, **kwargs) -> ArtifactStore:
        if isinstance(config, ArtifactStore):
            return config
        if isinstance(config, str):
            config = {"name": config, **kwargs}
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
    config: ArtifactStoreConfig

    def __init__(self, config: Any, **kwargs: Any) -> None:
        self.store = AutoArtifactStore(config, **kwargs)

    def __reduce__(self):
        return (self.__class__, (self.store.config,))

    def __repr__(self) -> str:
        p = str(self.store.config.to_dict()).replace("\n", " ")
        return f"KnowledgeBase({p})"

    def __getattr__(self, name: str) -> Any:
        try:
            return getattr(self.store, name)
        except AttributeError:
            return super().__getattr__(name)

    def extract_candidates(
        self, word: str | tuple[str, Any]
    ) -> Iterable[Candidate]:
        """Get all words and senses for a given form."""
        # get all the senses of the word
        sense_id = None
        if isinstance(word, tuple):
            word, sense_id = word
        if sense_id is not None:
            gloss = self.idx2gloss[sense_id]
            embedding = self.embeddings.get(sense_id, None)
            yield Candidate(word, gloss, embedding, sense_id).bind(self)
            return
        for sense_id in self.senses.get(word, None) or []:
            gloss = self.idx2gloss[sense_id]
            embedding = self.embeddings.get(sense_id, None)
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
                    # TODO: remove comment below if not needed
                    # if cc.word.lower() != form.lower():
                    #     continue
                    candidates.append(cc)
            yield Mention(form, Span(start, end), candidates)

    def augment(
        self, text: str, mentions=None, relation_filter=None
    ) -> Iterable[dict[str, Any]]:
        """Augment text by replacing words with synonyms, hyponyms, and hypernyms."""
        relation_filter = relation_filter or (lambda x: True)
        # not augmented
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
                    objects = self.triples.get(subject, {}).get(predicate, [])
                    for object_ in objects:
                        rel = Triple(subject, predicate, object_)
                        if not relation_filter(rel):
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

    def extract_relations(
        self, candidate: Candidate, predicates: Iterable[str] | None = None
    ) -> list[Triple]:
        if predicates is None:
            predicates = {
                "synonym",
                "antonym",
                "hypernym",
                "hyponym",
                "category",
            }
        subject = (candidate.word, candidate.sense_id)
        triples = []
        for predicate in set(predicates):
            objects = self.triples.get(subject, {}).get(predicate, [])
            for object_ in objects:
                rel = Triple(subject, predicate, object_)
                triples.append(rel)
        return triples


def extract_relations(
    kb: KnowledgeBase,
    candidate: Candidate,
    predicates: list[str] | None = None,
) -> list[Triple]:
    if predicates is None:
        predicates = ["synonym", "hyponym", "instance"]
    subject = (candidate.word, candidate.sense_id)
    triples = []
    for predicate in predicates:
        objects = kb.triples.get(subject, {}).get(predicate, [])
        for object_ in objects:
            rel = Triple(subject, predicate, object_)
            triples.append(rel)
    return triples


def get_candidate(kb: KnowledgeBase, word: str, sense_id: int) -> Candidate:
    """Retrieve a Candidate object for the given word and sense_id."""
    gloss = kb.idx2gloss[sense_id]
    embedding = kb.embeddings[sense_id]
    return Candidate(word, gloss, embedding, sense_id)
