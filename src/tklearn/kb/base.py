from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping
from typing import Any

import numpy as np

from tklearn.kb.lexicon import Lexicon
from tklearn.kb.models import Augmentation, Candidate, Mention, Span, Triple
from tklearn.kb.triple_store import TripleStore
from tklearn.utils.lang import get_stopwords
from tklearn.utils.registry import Registry

__all__ = [
    "KNOWLEDGE_STORES",
    "KnowledgeBase",
    "KnowledgeStore",
]

#: A word and an optional sense id.
WordSense = tuple[str, "int | None"]


class KnowledgeStore:
    """The data behind a `KnowledgeBase`.

    Subclasses load or build these attributes in their constructor.

    Attributes
    ----------
    lexicon : Lexicon
        Surface form -> set of ``(word, sense_id)`` it may refer to.
    triples : TripleStore
        Relations between ``(word, sense_id)`` pairs.
    gloss2idx, idx2gloss : Mapping
        Sense definitions (glosses) and their sense ids.
    senses : Mapping[str, set[int]]
        Word -> its sense ids.
    embeddings : Mapping[int, np.ndarray]
        Sense id -> embedding of its gloss.
    """

    lexicon: Lexicon[set[WordSense]]
    triples: TripleStore
    gloss2idx: Mapping[str, int]
    idx2gloss: Mapping[int, str]
    senses: Mapping[str, set[int]]
    embeddings: Mapping[int, np.ndarray]


#: Knowledge stores by name, e.g. ``KNOWLEDGE_STORES.create("wiktionary")``.
KNOWLEDGE_STORES: Registry[KnowledgeStore] = Registry("knowledge store")


class KnowledgeBase:
    """Find word senses in text and the relations between them.

    Parameters
    ----------
    store : KnowledgeStore or str, default="wiktionary"
        The store, or the name of a registered store.
    **kwargs
        Arguments for the store when it is given by name.

    Examples
    --------
    >>> kb = KnowledgeBase("wiktionary")
    >>> for mention in kb.extract_mentions("The dog barked."):
    ...     print(mention.form, len(mention.candidates))
    >>> for aug in kb.augment("The dog barked."):
    ...     print(aug.text, aug.relations)
    """

    def __init__(
        self, store: KnowledgeStore | str = "wiktionary", **kwargs: Any
    ) -> None:
        if isinstance(store, str):
            store = KNOWLEDGE_STORES.create(store, **kwargs)
        elif kwargs:
            msg = "keyword arguments are only used when 'store' is a name"
            raise TypeError(msg)
        self.store = store

    def __reduce__(self):
        # stores pickle as their constructor arguments, so a knowledge base
        # can be sent to worker processes without copying its data
        return (type(self), (self.store,))

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.store!r})"

    @property
    def lexicon(self) -> Lexicon[set[WordSense]]:
        return self.store.lexicon

    @property
    def triples(self) -> TripleStore:
        return self.store.triples

    @property
    def gloss2idx(self) -> Mapping[str, int]:
        return self.store.gloss2idx

    @property
    def idx2gloss(self) -> Mapping[int, str]:
        return self.store.idx2gloss

    @property
    def senses(self) -> Mapping[str, set[int]]:
        return self.store.senses

    @property
    def embeddings(self) -> Mapping[int, np.ndarray]:
        return self.store.embeddings

    def get_candidate(self, word: str, sense_id: int) -> Candidate:
        """Return the candidate for a known word sense."""
        gloss = self.idx2gloss[sense_id]
        embedding = self.embeddings.get(sense_id)
        return Candidate(word, gloss, embedding, sense_id)

    def extract_candidates(
        self, word: str | tuple[str, int | None]
    ) -> Iterator[Candidate]:
        """Yield the senses of a word, or the single sense it names.

        Parameters
        ----------
        word : str or (str, int or None)
            A word, or a ``(word, sense_id)`` pair. A pair with a sense id
            yields only that sense.
        """
        sense_id = None
        if isinstance(word, tuple):
            word, sense_id = word
        if sense_id is not None:
            yield self.get_candidate(word, sense_id)
            return
        for sense_id in self.senses.get(word) or []:
            yield self.get_candidate(word, sense_id)

    def extract_mentions(
        self,
        text: str,
        min_word_len: int = 3,
        stopwords: Iterable[str] | str = "english",
        exact_form: bool = False,
    ) -> Iterator[Mention]:
        """Find lexicon matches in a text with their candidate senses.

        Parameters
        ----------
        text : str
            The text to analyze.
        min_word_len : int, default=3
            Skip matches shorter than this.
        stopwords : iterable of str or str, default="english"
            Matches to skip, or the language of NLTK's stopword list.
        exact_form : bool, default=False
            Keep only candidates whose word equals the matched text
            (ignoring case). By default the lemmas of inflected forms are
            included too, e.g. "dog" for "dogs".
        """
        if isinstance(stopwords, str):
            stopwords = get_stopwords(stopwords)
        else:
            stopwords = {word.lower() for word in stopwords}
        for words, start, end in self.lexicon.extract(text):
            form = text[start:end]
            if len(form) < min_word_len or form.lower() in stopwords:
                continue
            candidates = [
                candidate
                for word in words
                for candidate in self.extract_candidates(word)
                if not exact_form or candidate.word.lower() == form.lower()
            ]
            yield Mention(form, Span(start, end), candidates)

    def extract_relations(
        self, candidate: Candidate, predicates: Iterable[str] | None = None
    ) -> list[Triple]:
        """Return the triples whose subject is the candidate's sense.

        Parameters
        ----------
        candidate : Candidate
            The subject.
        predicates : iterable of str, optional
            Only these relations (e.g. ``{"synonym", "hypernym"}``). By
            default, all of them.
        """
        subject = (candidate.word, candidate.sense_id)
        relations = self.triples.get(subject, {})
        if predicates is not None:
            predicates = set(predicates)
        return [
            Triple(subject, predicate, object_)
            for predicate, objects in relations.items()
            if predicates is None or predicate in predicates
            for object_ in objects
        ]

    def augment(
        self,
        text: str,
        mentions: Iterable[Mention] | None = None,
        predicates: Iterable[str] = ("synonym",),
        relation_filter: Callable[[Triple], bool] | None = None,
        include_original: bool = False,
    ) -> Iterator[Augmentation]:
        """Generate variants of a text by replacing mentions with related words.

        For each mention, every word related to one of its candidates by one
        of `predicates` yields one augmentation, supported by all the
        relations that lead to it.

        Parameters
        ----------
        text : str
            The text to augment.
        mentions : iterable of Mention, optional
            Mentions to replace; by default `extract_mentions(text)`.
        predicates : iterable of str, default=("synonym",)
            Relations used to find replacements.
        relation_filter : callable, optional
            Keep only relations for which ``relation_filter(triple)`` is true.
        include_original : bool, default=False
            First yield the unchanged text, with no relations.
        """
        predicates = set(predicates)
        if include_original:
            yield Augmentation(text, text, Span(0, 0), "", [])
        if mentions is None:
            mentions = self.extract_mentions(text)
        for mention in mentions:
            start, end = mention.span
            replacements: defaultdict[str, set[tuple]] = defaultdict(set)
            for candidate in mention.candidates:
                for triple in self.extract_relations(candidate, predicates):
                    if relation_filter is None or relation_filter(triple):
                        replacements[triple.object[0]].add(triple.to_tuple())
            for word, relations in replacements.items():
                yield Augmentation(
                    text=text[:start] + word + text[end:],
                    original=text,
                    span=mention.span,
                    replacement=word,
                    relations=sorted(relations, key=str),
                )
