import enum
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from typing import Generator, List, Optional

from sqlalchemy import (
    Column,
    Connection,
    Engine,
    Enum,
    Float,
    ForeignKey,
    MetaData,
    String,
    Table,
    create_engine,
)

session_var = ContextVar("session", default=None)

metadata = MetaData()


class Session:
    def __init__(self, engine: Engine):
        self.engine = engine
        self.connection = None

    @contextmanager
    def transaction(self) -> Generator[Connection, None, None]:
        in_transaction = self.connection is not None
        if not in_transaction:
            self.connection = self.engine.connect()
        if in_transaction:
            yield self.connection
        else:
            try:
                yield self.connection
                self.connection.commit()
            except Exception as e:
                self.connection.rollback()
                raise e
            finally:
                self.connection.close()
                self.connection = None


@contextmanager
def session(sess: Optional[Session] = None) -> Generator[Session, None, None]:
    if sess is None:
        token = session_var.set(sess)
        try:
            yield session_var.get()
        finally:
            session_var.reset(token)
    else:
        sess = session_var.get()
        if sess is None:
            msg = "no session available in context"
            raise ValueError(msg)
        yield sess


class NodeType(enum.Enum):
    CONCEPT = "concept"
    RELATION = "relation"


@dataclass
class Node:
    _id: str
    _type: NodeType
    label: str
    language: Optional[str]
    sense_label: Optional[str]
    term: Optional[str]
    site: Optional[str]
    site_available: Optional[bool]
    path: Optional[str]

    __table__ = Table(
        "node",
        metadata,
        Column("_id", String, primary_key=True),
        Column("_type", Enum(NodeType), nullable=False),
        Column("label", String, nullable=False),
        Column("language", String),
        Column("sense_label", String),
        Column("term", String),
        Column("site", String),
        Column("site_available", String),
        Column("path", String),
    )

    @classmethod
    def from_jsonld(cls, data: dict) -> "Node":
        return cls(
            _id=data["@id"],
            _type=NodeType(data["@type"]),
            label=data["label"],
            language=data.get("language"),
            sense_label=data.get("sense_label"),
            term=data.get("term"),
            site=data.get("site"),
            site_available=data.get("site_available"),
            path=data.get("path"),
        )


@dataclass
class Source:
    _id: str
    contributor: Optional[str]
    process: Optional[str]
    activity: Optional[str]

    __table__ = Table(
        "source",
        metadata,
        Column("_id", String, primary_key=True),
        Column("contributor", String),
        Column("process", String),
        Column("activity", String),
    )

    def from_jsonld(cls, data: dict) -> "Source":
        return cls(
            _id=data["@id"],
            contributor=data.get("contributor"),
            process=data.get("process"),
            activity=data.get("activity"),
        )


@dataclass
class Edge:
    _id: str
    _type: str
    rel_id: str
    start_id: str
    end_id: str
    surfaceText: Optional[str] = None
    sources: Optional[List[Source]] = None
    license: Optional[str] = None
    weight: Optional[float] = None
    dataset: Optional[str] = None

    __table__ = Table(
        "edge",
        metadata,
        Column("_id", String, primary_key=True),
        Column("_type", String, nullable=False),
        Column("rel_id", String, ForeignKey("node._id"), nullable=False),
        Column("start_id", String, ForeignKey("node._id"), nullable=False),
        Column("end_id", String, ForeignKey("node._id"), nullable=False),
        Column("surfaceText", String),
        Column("license", String),
        Column("weight", Float),
        Column("dataset", String),
    )

    @classmethod
    def from_jsonld(cls, data: dict) -> "Edge":
        return cls(
            _id=data["@id"],
            _type=data["@type"],
            rel_id=data["rel"]["@id"],
            start_id=data["start"]["@id"],
            end_id=data["end"]["@id"],
            surfaceText=data.get("surfaceText"),
            sources=[Source.from_jsonld(source) for source in data.get("sources", [])],
            license=data.get("license"),
            weight=data.get("weight"),
            dataset=data.get("dataset"),
        )


@dataclass
class EdgeSource:
    _id: str
    edge_id: str
    source_id: str

    __table__ = Table(
        "edge_source",
        metadata,
        Column("_id", String, primary_key=True),
        Column("edge_id", String, ForeignKey("edge._id"), nullable=False),
        Column("source_id", String, ForeignKey("source._id"), nullable=False),
    )

    @classmethod
    def from_jsonld(cls, data: dict) -> "EdgeSource":
        return cls(
            _id=data["@id"],
            edge_id=data["edge"]["@id"],
            source_id=data["source"]["@id"],
        )


class KnowledgeGraph:
    def __init__(self, url: str):
        self.session = Session(create_engine(url))
