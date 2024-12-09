from __future__ import annotations

import os
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Iterable, Union

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session
from tqdm import auto as tqdm

from tklearn import logging
from tklearn.config import config
from tklearn.core.triplet import TripletStore
from tklearn.core.vocab import Vocab
from tklearn.kb.loader import KnowledgeLoader
from tklearn.kb.models import Base, Edge, Node

__all__ = [
    "KnowledgeBase",
]

logger = logging.get_logger(__name__)


class KnowledgeBase:
    def __init__(
        self,
        database_name_or_path: str | Path,
        create: bool = False,
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
            # create the database if it does not exist
            if not input_path.exists() and create:
                input_engine = create_engine(
                    f"sqlite:///{input_path}", echo=verbose > 2
                )
                Base.metadata.create_all(input_engine)
        elif input_path.exists():
            database_name_or_path = os.path.join("default", input_path.stem)
        else:
            raise ValueError("invalid path")
        # output path with checksum
        output_path = cache_dir / f"{database_name_or_path}.db"
        self.path = output_path
        self.engine = create_engine(f"sqlite:///{self.path}", echo=verbose > 2)
        with self.session() as session:
            session.execute(text("PRAGMA journal_mode=WAL"))
        # create index if not exists
        self.index = self._get_or_create_index()

    def num_edges(self) -> int:
        with self.session() as session:
            return session.query(Edge).count()

    def _get_or_create_index(self) -> TripletStore:
        # populate index if not frozen (frozen means index is up-to-date)
        index_path = str(self.path.with_suffix("")) + "-index"
        index = TripletStore(index_path)
        if index.frozen:
            return index
        with self.session() as session:
            n_total = session.query(Edge).count()
            query = session.query(Edge.start_id, Edge.rel_id, Edge.end_id)
            pbar = tqdm.tqdm(
                query.yield_per(int(1e5)), total=n_total, desc="Indexing"
            )
            index.add(pbar)
        index.frozen = True
        return index

    @contextmanager
    def session(self) -> Generator[Session, None, None]:
        yield Session(self.engine)

    def cleanup(self):
        if not os.path.exists(self.path):
            return
        os.remove(self.path)

    def iternodes(self, verbose: bool = False) -> Iterable[Node]:
        with self.session() as session:
            query = session.query(Node)
            pbar = tqdm.tqdm(
                query.all(), desc="Iterating Nodes", disable=not verbose
            )
            for node in pbar:
                yield node

    def get_vocab(self) -> Vocab:
        with self.session() as session:
            n_total = session.query(Node).count()
        # populate vocab if not frozen (frozen means vocab is up-to-date)
        vocab_db_path = self.path.with_name(self.path.stem + "-vocab.db")
        config = {
            "type": "sqlalchemy",
            "url": f"sqlite:///{vocab_db_path}",
        }
        if os.path.exists(vocab_db_path):
            vocab = Vocab(config)
            return vocab
        vocab_db_temp_path = self.path.with_name(
            self.path.stem + "-vocab.db.tmp"
        )
        # remove the existing vocab database
        if os.path.exists(vocab_db_temp_path):
            os.remove(vocab_db_temp_path)
        tmp_config = {
            "type": "sqlalchemy",
            "url": f"sqlite:///{vocab_db_temp_path}",
        }
        vocab = Vocab(tmp_config)
        batch_size = int(1e4)
        with self.session() as session:
            query = session.query(Node)
            pbar = tqdm.tqdm(
                query.yield_per(batch_size),
                total=n_total,
                desc="Building Vocabulary",
            )
            nodes = []
            for i, node in enumerate(pbar):  # this will be done in parallel
                nodes.append(node)
                if i % batch_size == 0:
                    vocab.extend(nodes)
                    nodes = []
            if nodes:
                vocab.extend(nodes)
        del vocab
        # move the temporary database to the final location
        shutil.move(vocab_db_temp_path, vocab_db_path)
        vocab = Vocab(config)
        return vocab

    @classmethod
    def from_loader(cls, loader: KnowledgeLoader) -> KnowledgeBase:
        identifier = loader.config.identifier
        version = loader.config.version or "0.0.1"
        if not identifier.isidentifier():
            raise ValueError("identifier must be a valid Python identifier")
        database_name = os.path.join(identifier, f"{identifier}-v{version}")
        self = cls(database_name, create=True)
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
