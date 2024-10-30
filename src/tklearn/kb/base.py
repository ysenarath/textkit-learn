from __future__ import annotations

import os
import shutil
import urllib.parse
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, List, Tuple, Union

from rdflib import Graph, URIRef
from sqlalchemy import Index, create_engine, or_, text
from sqlalchemy.orm import Session
from tqdm import auto as tqdm

from tklearn.config import config
from tklearn.kb.loader import KnowledgeLoader
from tklearn.kb.models import Base, Edge, Node
from tklearn.utils import checksum as path_checksum

__all__ = [
    "KnowledgeBase",
]


# class KnowledgeGraph:
#     # index of the edges (s, p, o)
#     adj_list: dict[str, dict[str, set[str]]]

#     def __init__(self):
#         self.adj_list = {}

#     def add_edge(self, s: str, p: str, o: str):
#         if s not in self.adj_list:
#             self.adj_list[s] = {}
#         if p not in self.adj_list[s]:
#             self.adj_list[s][p] = set()
#         self.adj_list[s][p].add(o)

#     def get_edges(self, s: str, p: str) -> set[str]:
#         return self.adj_list.get(s, {}).get(p, set())


class KnowledgeBase:
    def __init__(
        self,
        database_name_or_path: str | Path,
        cache_dir: Union[Path, str, None] = None,
        verbose: int = 1,
    ):
        if cache_dir is None:
            cache_dir = Path(config.cache_dir) / "kb"
        else:
            cache_dir = Path(cache_dir)
        cache_dir = cache_dir.absolute().expanduser()
        input_path = Path(database_name_or_path)
        input_path = input_path.with_name(f"{input_path.name}.db")
        if not input_path.exists() and isinstance(database_name_or_path, str):
            input_path = cache_dir / f"{database_name_or_path}.db"
        elif input_path.exists():
            database_name_or_path = os.path.join("default", input_path.stem)
        else:
            raise ValueError("invalid path")
        # get checksum of the file
        checksum = path_checksum(input_path)
        # output path with checksum
        output_path = cache_dir / f"{database_name_or_path}-{checksum}.db"
        # copy the database file(s) to the cache directory
        for old_path in list(input_path.parent.glob(f"{input_path.name}*")):
            new_path_name = f"{old_path.stem}-{checksum}{old_path.suffix}"
            new_path = output_path.with_name(new_path_name)
            if new_path.exists():
                continue
            shutil.copy(old_path, new_path)
        if path_checksum(output_path) != checksum:
            raise ValueError("invalid checksum")
        self.path = output_path
        self.engine = create_engine(f"sqlite:///{self.path}", echo=verbose > 2)
        Base.metadata.create_all(self.engine)
        with self.session() as session:
            session.execute(text("PRAGMA journal_mode=WAL"))

    def import_from_loader(self, loader: KnowledgeLoader):
        identifier = loader.config.identifier
        if not identifier.isidentifier():
            raise ValueError("identifier must be a valid Python identifier")
        with self.session() as session:
            session.execute(text("PRAGMA journal_mode=WAL"))
            for i, val in enumerate(loader.iterrows()):
                _ = Edge.from_dict(  # ignore the return value
                    val,
                    session=session,
                    commit=False,
                    namespace=loader.config.namespace,
                )
                if i % 100 == 0:
                    try:
                        session.commit()
                    except Exception as e:
                        session.rollback()
                        raise e
            try:
                session.commit()
            except Exception as e:
                session.rollback()
                raise e

    @contextmanager
    def session(self) -> Generator[Session, None, None]:
        yield Session(self.engine)

    def cleanup(self):
        if not os.path.exists(self.path):
            return
        os.remove(self.path)

    def get_triples(self) -> List[Tuple[str, str, str]]:
        with self.session() as session:
            triples: List[Tuple[str, str, str]] = session.query(
                Edge.start_id, Edge.rel_id, Edge.end_id
            ).all()
        return triples

    def get_vocab(self) -> List[str]:
        with self.session() as session:
            vocab: List[str] = session.query(Node.id, Node.label).all()
        return vocab

    def find_neighbors(
        self, node_ids: List[str | Node] | str | Node
    ) -> List[str]:
        if isinstance(node_ids, Node):
            node_ids = node_ids.id
        if isinstance(node_ids, str):
            node_ids = [node_ids]
        node_ids = [
            node_id if isinstance(node_id, str) else node_id.id
            for node_id in node_ids
        ]
        with self.session() as session:
            neighbors: List[str] = (
                session.query(Edge.start_id, Edge.rel_id, Edge.end_id)
                .filter(
                    or_(
                        Edge.start_id.in_(node_ids),
                        Edge.end_id.in_(node_ids),
                    )
                )
                .all()
            )
        return neighbors

    def prepare(self):
        edge_node_idse_index = Index(
            "ix_edge_node_ids", Edge.start_id, Edge.end_id
        )
        edge_node_idse_index.create(bind=self.engine)

    def create_graph(self) -> Graph:
        graph = Graph()
        with self.session() as session:
            n_total = session.query(Edge).count()
            query = session.query(Edge.start_id, Edge.rel_id, Edge.end_id)
            pbar = tqdm.tqdm(total=n_total, desc="Creating graph")
            for triple in query.yield_per(1000):
                triple = tuple(map(uriref, triple))
                graph.add(triple)
                pbar.update(1)
        return graph


def uriref(val: str) -> URIRef:
    val = urllib.parse.quote(val)
    return URIRef(val)
