from typing import Iterable

from tklearn.core.model import Span

__all__ = [
    "KnowledgeBase",
]


class KnowledgeBase:
    """A `KnowledgeBase` instance stores unique identifiers for entities and
    their textual aliases, to support entity linking of named entities to
    real-world concepts.
    This is an abstract class and requires its operations to be implemented.
    """

    def __init__(self):
        """Create a KnowledgeBase."""
        # Make sure abstract KB is not instantiated.
        if self.__class__ == KnowledgeBase:
            raise TypeError()

    def get_candidates(self, mention: Span) -> Iterable[str]:
        """
        Return candidate entities for specified text. Each candidate defines
        the entity, the original alias,
        and the prior probability of that alias resolving to that entity.
        If the no candidate is found for a given text, an empty list is returned.
        mention (Span): Mention for which to get candidates.
        RETURNS (Iterable[Candidate]): Identified candidates.
        """
        raise NotImplementedError

    def get_candidates_batch(self, mentions: Iterable[Span]) -> Iterable[Iterable[str]]:
        """
        Return candidate entities for specified texts. Each candidate defines
        the entity, the original alias, and the prior probability of that
        alias resolving to that entity.
        If no candidate is found for a given text, an empty list is returned.
        mentions (Iterable[Span]): Mentions for which to get candidates.
        RETURNS (Iterable[Iterable[Candidate]]): Identified candidates.
        """
        return [self.get_candidates(span) for span in mentions]

    def get_vectors(self, entities: Iterable[str]) -> Iterable[Iterable[float]]:
        """
        Return vectors for entities.
        entity (str): Entity name/ID.
        RETURNS (Iterable[Iterable[float]]): Vectors for specified entities.
        """
        return [self.get_vector(entity) for entity in entities]

    def get_vector(self, entity: str) -> Iterable[float]:
        """
        Return vector for entity.
        entity (str): Entity name/ID.
        RETURNS (Iterable[float]): Vector for specified entity.
        """
        raise NotImplementedError
