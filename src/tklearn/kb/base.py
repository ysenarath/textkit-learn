from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator

import numpy as np

from tklearn.kb.models import Augmentation, Candidate, Mention, Span, Triple
from tklearn.utils.lang import get_stopwords

__all__ = [
    "INVERSE_PREDICATES",
    "KnowledgeBase",
    "KnowledgeStore",
]

#: A word and an optional sense id.
WordSense = tuple[str, "int | None"]

#: How a relation reads from its object's side: ``x p y`` gives ``y q x``.
INVERSE_PREDICATES = {
    "synonym": "synonym",
    "antonym": "antonym",
    "hypernym": "hyponym",
    "hyponym": "hypernym",
}


class KnowledgeStore(ABC):
    """The data behind a `KnowledgeBase`.

    A sense id identifies a gloss. A sense id of None means the word in
    general, e.g. a relation that holds for every sense of a word.
    """

    @abstractmethod
    def find_forms(
        self, text: str
    ) -> Iterator[tuple[int, int, list[WordSense]]]:
        """Yield ``(start, end, word_senses)`` for each known form in a text.

        Matches may overlap, e.g. "at the bank" and "bank".
        """

    @abstractmethod
    def get_gloss(self, sense_id: int) -> str:
        """Return the gloss of a sense; raise KeyError if it is unknown."""

    @abstractmethod
    def get_senses(self, word: str) -> list[int]:
        """Return the sense ids of a word."""

    @abstractmethod
    def get_relations(
        self,
        word: str,
        sense_id: int | None = None,
        predicates: Iterable[str] | None = None,
        by: str = "subject",
    ) -> list[Triple]:
        """Return the stored triples with a word as subject or object.

        Parameters
        ----------
        word : str
            The word to look up.
        sense_id : int, optional
            Keep triples where the word has this sense or no sense. By
            default, all of them.
        predicates : iterable of str, optional
            Only these relations. By default, all of them.
        by : {"subject", "object"}, default="subject"
            The side of the triple the word is on.
        """

    def get_embedding(self, sense_id: int) -> np.ndarray | None:
        """Return the embedding of a sense's gloss, or None."""
        return None


class KnowledgeBase:
    """Find word senses in text and the relations between them.

    Parameters
    ----------
    store : KnowledgeStore
        The data to query, e.g. a `WiktionaryStore`.

    Examples
    --------
    >>> kb = KnowledgeBase(WiktionaryStore())
    >>> for mention in kb.extract_mentions("The dog barked."):
    ...     print(mention.form, len(mention.candidates))
    >>> for aug in kb.augment("The dog barked."):
    ...     print(aug.text, aug.relations)
    """

    def __init__(self, store: KnowledgeStore) -> None:
        if not isinstance(store, KnowledgeStore):
            msg = (
                "expected a KnowledgeStore such as WiktionaryStore(), got "
                f"{type(store).__name__}"
            )
            raise TypeError(msg)
        self.store = store

    def __reduce__(self):
        # stores pickle as their constructor arguments, so a knowledge base
        # can be sent to worker processes without copying its data
        return (type(self), (self.store,))

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.store!r})"

    def get_candidate(self, word: str, sense_id: int) -> Candidate:
        """Return the candidate for a known word sense."""
        gloss = self.store.get_gloss(sense_id)
        embedding = self.store.get_embedding(sense_id)
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
        for sense_id in self.store.get_senses(word):
            yield self.get_candidate(word, sense_id)

    def extract_mentions(
        self,
        text: str,
        min_word_len: int = 3,
        stopwords: Iterable[str] | str = "english",
        exact_form: bool = False,
    ) -> Iterator[Mention]:
        """Find known forms in a text with their candidate senses.

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
        for start, end, words in self.store.find_forms(text):
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

        These include the relations of the word with no sense, and those
        stated from the other word's side (see `INVERSE_PREDICATES`), e.g.
        ``puppy hypernym dog`` gives ``dog hyponym puppy``. A relation
        found more than once is returned once, with the most senses known.

        Parameters
        ----------
        candidate : Candidate
            The subject.
        predicates : iterable of str, optional
            Only these relations (e.g. ``{"synonym", "hypernym"}``). By
            default, all of them.
        """
        word, sense_id = candidate.word, candidate.sense_id
        if predicates is not None:
            predicates = set(predicates)
        triples = self.store.get_relations(word, sense_id, predicates)
        # stored predicate -> the predicate it gives from the object's side
        inverses = {
            stored: predicate
            for predicate, stored in INVERSE_PREDICATES.items()
            if predicates is None or predicate in predicates
        }
        if inverses:
            for subject, stored, object_ in self.store.get_relations(
                word, sense_id, inverses, by="object"
            ):
                triples.append(Triple(object_, inverses[stored], subject))
        best: dict[tuple[str, str], Triple] = {}
        for triple in triples:
            key = (triple.predicate, triple.object[0])
            if key not in best or _num_senses(triple) > _num_senses(best[key]):
                best[key] = triple
        return list(best.values())

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
        relations that lead to it. A word equal to the mention, ignoring
        case, is skipped: it would give back the same text.

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
            form = text[start:end].lower()
            replacements: defaultdict[str, set[tuple]] = defaultdict(set)
            for candidate in mention.candidates:
                for triple in self.extract_relations(candidate, predicates):
                    if relation_filter is None or relation_filter(triple):
                        replacements[triple.object[0]].add(triple.to_tuple())
            for word, relations in replacements.items():
                if word.lower() == form:
                    continue
                yield Augmentation(
                    text=text[:start] + word + text[end:],
                    original=text,
                    span=mention.span,
                    replacement=word,
                    relations=sorted(relations, key=str),
                )


def _num_senses(triple: Triple) -> int:
    return (triple.subject[1] is not None) + (triple.object[1] is not None)
