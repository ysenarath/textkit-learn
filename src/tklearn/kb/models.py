from __future__ import annotations

import weakref
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, NamedTuple, TypeVar

import numpy as np
from typing_extensions import Self

T = TypeVar("T")


@dataclass(frozen=True, order=True)
class Span:
    start: int
    end: int

    def __iter__(self) -> Iterable[int]:
        yield self.start
        yield self.end


@dataclass(order=True)
class Mention:
    form: str
    span: Span
    candidates: list[Candidate]


@dataclass(order=True)
class Candidate:
    word: str
    definition: str
    embedding: np.ndarray | None
    sense_id: int

    def bind(self, wordex: Any) -> Candidate:
        setattr(self, "_wordex", weakref.ref(wordex))
        return self

    @property
    def wordex(self) -> Any:
        get_wordex = getattr(self, "_wordex", None)
        if get_wordex is None:
            raise AttributeError("Candidate is not bound to a Wordex instance")
        return get_wordex()


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
