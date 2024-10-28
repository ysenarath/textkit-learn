from tklearn.kb.base import KnowledgeBase
from tklearn.kb.conceptnet import ConceptNetLoader
from tklearn.kb.loader import KnowledgeLoader

__all__ = [
    "KnowledgeBase",
    "KnowledgeLoader",
]

# add loaders here so the import is not removed by isort
_ConceptNetLoader = ConceptNetLoader
