from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import NamedTuple, TypeVar

import numpy as np
from typing_extensions import Self

__all__ = [
    "Augmentation",
    "Candidate",
    "Mention",
    "Span",
    "Triple",
]

T = TypeVar("T")


@dataclass(frozen=True, order=True)
class Span:
    """Character offsets ``[start, end)`` in a text."""

    start: int
    end: int

    def __iter__(self) -> Iterable[int]:
        yield self.start
        yield self.end


@dataclass(order=True)
class Mention:
    """A lexicon match in a text and the senses it may refer to."""

    form: str
    span: Span
    candidates: list[Candidate]

    def __repr__(self):
        return f"Mention(form={self.form!r}, span=({self.span.start}, {self.span.end}), candidates={self.candidates})"


@dataclass(order=True)
class Candidate:
    """One sense of a word that a mention may refer to."""

    word: str
    definition: str
    embedding: np.ndarray | None
    sense_id: int

    def __repr__(self):
        return f"Candidate(word={self.word!r}, sense_id={self.sense_id}, definition={self.definition!r})"


def concept2tuple(
    concept: str | tuple[str, int | None],
) -> tuple[str, int | None]:
    if isinstance(concept, str) or concept is None:
        concept = (concept, None)
    elif len(concept) == 1:
        concept = (*concept, None)
    concept_word, concept_sense = concept
    if concept_sense and concept_sense < 0:
        concept_sense = None
    return concept_word, concept_sense


class Triple(NamedTuple):
    subject: tuple[T, int | None]
    predicate: str
    object: tuple[T, int | None]

    @staticmethod
    def from_tuple(triple: tuple) -> Self:
        if len(triple) < 3:
            # make sure we have three elements
            triple = triple + (None,) * (3 - len(triple))
        elif len(triple) == 5:
            # if there are five elements, we assume the first two and last two
            # are the subject and object respectively encoded as tuples
            # (subject_word, subject_sense, predicate, object_word, object_sense)
            # where subject_word and object_word are the words and
            # subject_sense and object_sense are the senses (type int)
            # so we convert them to tuples of (word, sense)
            triple = (triple[0], triple[1]), triple[2], (triple[3], triple[4])
        subject, predicate, object_ = triple
        subject = concept2tuple(subject)
        object_ = concept2tuple(object_)
        return Triple(subject, predicate, object_)

    def to_tuple(self) -> tuple:
        if len(self) < 3:
            self = self + (None,) * (3 - len(self))
        try:
            subject, predicate, object = self
        except ValueError:
            raise ValueError(
                f"triple must have exactly three elements, got {len(self)}"
            )
        if isinstance(subject, str) or subject is None:
            subject = (subject, None)
        elif len(subject) == 1:
            subject = (*subject, None)
        subject_word, subject_sense = subject
        if subject_sense is None or subject_sense < 0:
            subject_sense = -1
        if isinstance(object, str) or object is None:
            object = (object, None)
        elif len(object) == 1:
            object = (*object, None)
        object_word, object_sense = object
        if object_sense is None or object_sense < 0:
            object_sense = -1
        return (
            subject_word,
            subject_sense,
            predicate,
            object_word,
            object_sense,
        )


@dataclass
class Augmentation:
    """A text with one mention replaced by a related word.

    Attributes
    ----------
    text : str
        The augmented text.
    original : str
        The text it was derived from.
    span : Span
        Character offsets of the replaced mention in `original`.
    replacement : str
        The word that replaced the mention.
    relations : list of tuple
        The triples ``(subject_word, subject_sense, predicate, object_word,
        object_sense)`` that link the mention to `replacement`.
    """

    text: str
    original: str
    span: Span
    replacement: str
    relations: list[tuple] = field(default_factory=list)

    @property
    def support(self) -> int:
        """Number of relations supporting the replacement."""
        return len(self.relations)
