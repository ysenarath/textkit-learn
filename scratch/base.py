from __future__ import annotations

from typing import NamedTuple

import rdflib
from rdflib.term import Node
from typing_extensions import Self

__all__ = [
    "KnowledgeBase",
]


class Triple(NamedTuple):
    subject: Node
    predicate: Node
    object: Node


class KnowledgeBase(rdflib.Graph):
    def add(self, triple: Triple) -> Self:
        """Add a triple with self as context"""
        return super().add(triple)
