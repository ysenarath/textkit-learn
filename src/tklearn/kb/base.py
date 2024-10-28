import os
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Union

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from tklearn.config import config
from tklearn.kb.loader import KnowledgeLoader
from tklearn.kb.models import Base, Edge

__all__ = [
    "KnowledgeBase",
]


class KnowledgeBase:
    def __init__(self, path: Union[Path, str, None] = None, verbose: int = 1):
        if path is None:
            path = Path(config.cache_dir) / "kb"
            path = path.absolute().expanduser()
        else:
            path = Path(path).absolute().expanduser()
        self.path = path
        self.path.mkdir(parents=True, exist_ok=True)
        uri = f"sqlite:///{self.path / 'base' / 'data.db'}"
        self.engine = create_engine(uri, echo=verbose > 2)
        Base.metadata.create_all(self.engine)

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
