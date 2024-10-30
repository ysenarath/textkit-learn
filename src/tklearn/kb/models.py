from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from rdflib import RDF, Literal, URIRef
from sqlalchemy import ForeignKey
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    relationship,
)
from typing_extensions import Self

from tklearn.kb.conceptnet import CN


def apply_prefix(uri: Optional[str], namespace: Optional[str] = None) -> str:
    if uri is None:
        return None
    if namespace and uri.startswith("/"):
        namespace = namespace.rstrip("/")
        return f"{namespace}{uri}"
    return uri


class Base(DeclarativeBase):
    pass


class Node(Base):
    __tablename__ = "node"

    id: Mapped[str] = mapped_column(primary_key=True)
    label: Mapped[Optional[str]] = mapped_column(nullable=True)
    language: Mapped[Optional[str]]
    sense_label: Mapped[Optional[str]]
    term_id: Mapped[Optional[str]] = mapped_column(ForeignKey("node.id"))
    site: Mapped[Optional[str]]
    path: Mapped[Optional[str]]
    site_available: Mapped[Optional[bool]]

    term: Mapped[Optional["Node"]] = relationship()

    @classmethod
    def from_dict(
        cls,
        data: dict,
        session: Optional[Session] = None,
        commit: bool = True,
        namespace: Optional[str] = None,
    ) -> Self:
        id_ = apply_prefix(data["@id"], namespace=namespace)
        instance = session.get(cls, id_)
        if instance:
            return instance
        term_id = data.get("term")
        if term_id:
            term_id = apply_prefix(term_id, namespace=namespace)
        instance = cls(
            id=id_,
            label=data["label"],
            language=data.get("language"),
            sense_label=data.get("sense_label"),
            term_id=term_id,
            site=data.get("site"),
            path=data.get("path"),
            site_available=data.get("site_available"),
        )
        try:
            session.add(instance)
            if commit:
                session.commit()
            return instance
        except IntegrityError as e:
            session.rollback()
            # If we get an integrity error, someone else beat us to it
            # Try one more time to get the instance
            instance = session.get(cls, id_)
            if instance:
                return instance
            # If we still don't have an instance, something else went wrong
            raise e from None

    def to_triples(self) -> List[Tuple]:
        triples = []
        node_uri = URIRef(CN[self.id])
        triples.append((node_uri, RDF.type, CN.Concept))
        triples.append((node_uri, CN.label, Literal(self.label)))
        if self.language:
            triples.append((node_uri, CN.language, Literal(self.language)))
        if self.term:
            triples.append((node_uri, CN.term, URIRef(CN[self.term.id])))
        if self.sense_label:
            triples.append((
                node_uri,
                CN.sense_label,
                Literal(self.sense_label),
            ))
        if self.site:
            triples.append((node_uri, CN.site, Literal(self.site)))
        if self.path:
            triples.append((node_uri, CN.path, Literal(self.path)))
        if self.site_available:
            triples.append((
                node_uri,
                CN.site_available,
                Literal(self.site_available),
            ))
        return triples

    def __repr__(self) -> str:
        return f"Node(id='{self.id}', label='{self.label}', language='{self.language}')"


class Relation(Base):
    __tablename__ = "relation"

    id: Mapped[str] = mapped_column(primary_key=True)
    label: Mapped[str]
    symmetric: Mapped[bool] = mapped_column(default=False)

    @classmethod
    def from_dict(
        cls,
        data: dict,
        session: Optional[Session] = None,
        commit: bool = True,
        namespace: Optional[str] = None,
    ) -> Self:
        id_ = apply_prefix(data["@id"], namespace=namespace)
        instance = session.get(cls, id_)
        if instance:
            return instance
        instance = cls(
            id=id_,
            label=data["label"],
            symmetric=data.get("symmetric", False),
        )
        try:
            session.add(instance)
            if commit:
                session.commit()
            return instance
        except IntegrityError as e:
            session.rollback()
            # If we get an integrity error, someone else beat us to it
            # Try one more time to get the instance
            instance = session.get(cls, id_)
            if instance:
                return instance
            # If we still don't have an instance, something else went wrong
            raise e from None

    def to_triples(self) -> List[Tuple]:
        relation_uri = URIRef(CN[self.id])
        triples = [(relation_uri, RDF.type, CN.Relation)]
        triples.append((relation_uri, CN.label, Literal(self.label)))
        triples.append((relation_uri, CN.symmetric, Literal(self.symmetric)))
        return triples

    def __repr__(self) -> str:
        return f"Relation(id='{self.id}', label='{self.label}', symmetric={self.symmetric})"


class Source(Base):
    __tablename__ = "source"

    id: Mapped[str] = mapped_column(primary_key=True)
    contributor: Mapped[Optional[str]]
    process: Mapped[Optional[str]]
    activity: Mapped[Optional[str]]
    edge_id: Mapped[str] = mapped_column(ForeignKey("edge.id"))

    @classmethod
    def from_dict(
        cls,
        data: dict,
        session: Optional[Session] = None,
        commit: bool = True,
        namespace: Optional[str] = None,
    ) -> Self:
        id_ = apply_prefix(data["@id"], namespace=namespace)
        edge_id = data["edge_id"]  # assume this is already prefixed
        instance = session.get(cls, id_)
        if instance:
            return instance
        kwargs = {}
        for key in ("contributor", "process", "activity"):
            if key not in data:
                continue
            value = data[key]
            kwargs[key] = apply_prefix(value, namespace=namespace)
        instance = cls(id=id_, edge_id=edge_id, **kwargs)
        try:
            session.add(instance)
            if commit:
                session.commit()
            return instance
        except IntegrityError as e:
            session.rollback()
            # If we get an integrity error, someone else beat us to it
            # Try one more time to get the instance
            instance = session.get(cls, id_)
            if instance:
                return instance
            # If we still don't have an instance, something else went wrong
            raise e from None

    def to_triples(self) -> List[Tuple]:
        source_uri = URIRef(CN[self.id])
        triples = [(source_uri, RDF.type, CN.Source)]
        if self.contributor:
            triples.append((
                source_uri,
                CN.contributor,
                Literal(self.contributor),
            ))
        if self.process:
            triples.append((source_uri, CN.process, Literal(self.process)))
        if self.activity:
            triples.append((source_uri, CN.activity, Literal(self.activity)))
        return triples

    def __repr__(self) -> str:
        return f"Source(id='{self.id}', contributor='{self.contributor}')"


class Edge(Base):
    __tablename__ = "edge"

    id: Mapped[str] = mapped_column(primary_key=True)
    rel_id: Mapped[str] = mapped_column(ForeignKey("relation.id"), index=True)
    start_id: Mapped[str] = mapped_column(ForeignKey("node.id"), index=True)
    end_id: Mapped[str] = mapped_column(ForeignKey("node.id"), index=True)
    license: Mapped[Optional[str]]
    weight: Mapped[float] = mapped_column(default=1.0)
    dataset: Mapped[Optional[str]]
    surface_text: Mapped[Optional[str]]

    rel: Mapped[Relation] = relationship()
    start: Mapped[Node] = relationship(foreign_keys=[start_id])
    end: Mapped[Node] = relationship(foreign_keys=[end_id])
    sources: Mapped[List[Source]] = relationship()

    @classmethod
    def from_dict(
        cls,
        data: dict,
        session: Optional[Session] = None,
        commit: bool = True,
        namespace: Optional[str] = None,
    ) -> Self:
        id_ = apply_prefix(data["@id"], namespace=namespace)
        instance = session.get(cls, id_)
        if instance:
            return instance
        rel = Relation.from_dict(
            data["rel"], session=session, commit=commit, namespace=namespace
        )
        start = Node.from_dict(
            data["start"], session=session, commit=commit, namespace=namespace
        )
        end = Node.from_dict(
            data["end"], session=session, commit=commit, namespace=namespace
        )
        sources = [
            Source.from_dict(
                {"edge_id": id_, **source},
                session=session,
                commit=commit,
                namespace=namespace,
            )
            for source in data["sources"]
        ]
        dataset = apply_prefix(data.get("dataset"), namespace=namespace)
        instance = cls(
            id=id_,
            rel=rel,
            start=start,
            end=end,
            sources=sources,
            license=data["license"],
            weight=data["weight"],
            dataset=dataset,
            surface_text=data.get("surfaceText", data.get("surface_text")),
        )
        try:
            session.add(instance)
            if commit:
                session.commit()
            return instance
        except IntegrityError as e:
            session.rollback()
            # If we get an integrity error, someone else beat us to it
            # Try one more time to get the instance
            instance = session.get(cls, id_)
            if instance:
                return instance
            # If we still don't have an instance, something else went wrong
            raise e from None

    def to_triples(self) -> List[Tuple]:
        start_node = URIRef(CN[self.start.id])
        end_node = URIRef(CN[self.end.id])
        relation = URIRef(CN[self.rel.id])
        edge_uri = URIRef(CN[self.id])

        triples = []

        # add edge
        triples.append((edge_uri, RDF.type, CN.Assertion))
        triples.append((edge_uri, CN.relation, relation))
        triples.append((edge_uri, CN.start, start_node))
        triples.append((edge_uri, CN.end, end_node))
        triples.append((edge_uri, CN.weight, Literal(self.weight)))

        if self.dataset:
            triples.append((
                edge_uri,
                CN.dataset,
                Literal(self.dataset),
            ))
        if self.surface_text:
            triples.append((
                edge_uri,
                CN.surfaceText,
                Literal(self.surface_text),
            ))
        if self.license:
            triples.append((
                edge_uri,
                CN.license,
                Literal(self.license),
            ))

        if self.sources:
            for source in self.sources:
                source_uri = URIRef(CN[source.id])
                triples.append((edge_uri, CN.sources, source_uri))
        return triples

    def __repr__(self) -> str:
        return f"Edge(id='{self.id}', start='{self.start_id}', rel='{self.rel_id}', end='{self.end_id}')"


@dataclass
class Feature:
    rel: Optional[Relation] = None
    start: Optional[Node] = None
    end: Optional[Node] = None
    node: Optional[Node] = None


@dataclass
class RelatedNode:
    id: str
    weight: float


@dataclass
class PartialCollectionView:
    paginatedProperty: str
    firstPage: str
    nextPage: Optional[str] = None
    previousPage: Optional[str] = None


@dataclass
class Query:
    id: str
    edges: Optional[List[Edge]] = None
    features: Optional[List[Feature]] = None
    related: Optional[List[RelatedNode]] = None
    view: Optional[PartialCollectionView] = None
    value: Optional[float] = None
    license: Optional[str] = None
