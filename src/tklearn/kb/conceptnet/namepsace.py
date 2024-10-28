from rdflib.namespace import DefinedNamespace, Namespace
from rdflib.term import URIRef

__all__ = [
    "CN",
]


class CN(DefinedNamespace):
    _warn = False
    _fail = False

    Concept: URIRef
    label: URIRef
    language: URIRef
    sense_label: URIRef
    term: URIRef
    site: URIRef
    path: URIRef
    site_available: URIRef

    Relation: URIRef
    symmetric: URIRef
    # label: URIRef

    Assertion: URIRef
    relation: URIRef
    start: URIRef
    end: URIRef
    sources: URIRef
    license: URIRef
    weight: URIRef
    dataset: URIRef
    surfaceText: URIRef

    Source: URIRef
    contributor: URIRef
    process: URIRef
    activity: URIRef

    _NS = Namespace("http://conceptnet.io/")

    def __class_getitem__(cls, key: str) -> URIRef:
        if key.startswith("/"):
            key = key[1:]
        return super().__class_getitem__(key)

    def __getattr__(cls, key: str) -> URIRef:
        return cls[key]
