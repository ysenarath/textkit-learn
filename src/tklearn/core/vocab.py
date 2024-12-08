from __future__ import annotations

import threading
from dataclasses import MISSING
from typing import (
    Any,
    Dict,
    Generator,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Tuple,
    TypeVar,
)

import numpy as np
import torch
from nightjar import BaseConfig
from sqlalchemy import (
    Column,
    Index,
    Integer,
    String,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, scoped_session, sessionmaker
from sqlalchemy.pool import QueuePool
from tqdm import auto as tqdm

T = TypeVar("T", List[int], np.ndarray, torch.Tensor)


class Base(DeclarativeBase):
    pass


class VocabToken(Base):
    """SQLAlchemy model for vocabulary tokens."""

    __tablename__ = "vocabulary_tokens"

    id = Column(Integer, primary_key=True)
    token = Column(String(256), unique=True, nullable=False)

    # Indexes for frequent queries
    __table_args__ = (Index("idx_token", "token"),)


class VocabItem(NamedTuple):
    id: int
    token: str


class VocabConfig(BaseConfig, dispatch=["type"]):
    type: str


class VocabStore:
    def add(self, token: str, index: int | None = None) -> int:
        pass

    def get_index(
        self, token: str, default: int | None = MISSING
    ) -> int | None:
        pass

    def get_token(
        self, index: int, default: str | None = MISSING
    ) -> str | None:
        pass

    def __len__(self) -> int:
        pass

    def items(self) -> Generator[Tuple[VocabItem], None, None]:
        pass


class SQLAlchemyVocabConfig(VocabConfig):
    type: str = "sqlalchemy"
    url: Optional[str] = None


class SQLAlchemyVocabStore:
    """SQLAlchemy-based persistent vocabulary implementation with advanced features."""

    def __init__(
        self, url: str, cache_size: int = 10000, verbose: bool = False
    ):
        self.db_url = url
        self.cache_size = cache_size
        # Initialize SQLAlchemy engine with connection pooling
        self.engine = create_engine(
            url,
            poolclass=QueuePool,
            pool_size=5,
            max_overflow=10,
            pool_timeout=30,
            echo=verbose,
        )
        # Create tables
        Base.metadata.create_all(self.engine)
        # Create session factory
        self.Session = scoped_session(sessionmaker(bind=self.engine))
        # Initialize token cache
        self._token_cache: Dict[str, int] = {}
        self._index_cache: Dict[int, str] = {}
        self._cache_lock = threading.Lock()

    def add(self, token: str, index: int | None = None) -> int:
        session = self.Session()
        try:
            # Check if token exists
            vocab_token = (
                session.query(VocabToken).filter_by(token=token).first()
            )
            if vocab_token:
                token_id = vocab_token.id
            else:
                # Create new token
                vocab_token = VocabToken(token=token, id=index)
                session.add(vocab_token)
                session.flush()  # Get ID without committing
                token_id = vocab_token.id
            session.commit()
            # Update cache
            with self._cache_lock:
                self._token_cache[token] = token_id
                self._index_cache[token_id] = token
            return token_id
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def extend(self, tokens: Iterable[str | Tuple[int, str]]) -> None:
        # only commit once
        session = self.Session()
        try:
            for token in tokens:
                if isinstance(token, str):
                    self.add(token)
                else:
                    self.add(token[1], index=token[0])
            session.commit()
        except Exception as e:
            session.rollback()
            raise e

    def get_index(
        self, token: str, default: int | None = MISSING
    ) -> int | None:
        """Get index for a token."""
        # Check cache first
        with self._cache_lock:
            if token in self._token_cache:
                return self._token_cache[token]
        session = self.Session()
        try:
            vocab_token = (
                session.query(VocabToken).filter_by(token=token).first()
            )
            if vocab_token:
                # Update cache and return
                with self._cache_lock:
                    self._token_cache[token] = vocab_token.id
                    self._index_cache[vocab_token.id] = token
                return vocab_token.id
            if default is not MISSING:
                return default
            raise KeyError(f"token '{token}' not found")
        finally:
            session.close()

    def get_token(
        self, index: int, default: str | None = MISSING
    ) -> str | None:
        """Get token for an index."""
        # Check cache first
        with self._cache_lock:
            if index in self._index_cache:
                return self._index_cache[index]
        session = self.Session()
        try:
            vocab_token = session.query(VocabToken).filter_by(id=index).first()
            if vocab_token:
                # Update cache and return
                with self._cache_lock:
                    self._token_cache[vocab_token.token] = index
                    self._index_cache[index] = vocab_token.token
                return vocab_token.token
            if default is not MISSING:
                return default
            raise IndexError(f"index '{index}' not found")
        finally:
            session.close()

    def __len__(self) -> int:
        """Get vocabulary size."""
        session = self.Session()
        try:
            return session.query(VocabToken).count()
        finally:
            session.close()

    def items(
        self, verbose: bool = False
    ) -> Generator[Tuple[VocabItem], None, None]:
        """Iterate over all tokens in the vocabulary."""
        session = self.Session()
        try:
            q = session.query(VocabToken)
            n = q.count()
            for vocab_token in tqdm.tqdm(
                q.yield_per(1000), total=n, disable=not verbose
            ):
                yield VocabItem(vocab_token.id, vocab_token.token)
        finally:
            session.close()


class Vocab:
    def __init__(self, config: VocabConfig | dict):
        """Initialize the triplet store with a path to the LevelDB database."""
        self.config = (
            config
            if isinstance(config, VocabConfig)
            else VocabConfig.from_dict(config)
        )
        self.store = None
        self.open()

    def open(self):
        """Open the database connection."""
        if isinstance(self.config, SQLAlchemyVocabConfig):
            self.store = SQLAlchemyVocabStore(self.config.url)
        else:
            raise ValueError("unsupported vocab store type")
        return self

    def get_index(self, text: str) -> int:
        try:
            default = self.store.get_index("[UNK]")
        except KeyError:
            return self.store.get_index(text)
        return self.store.get_index(text, default=default)

    def get_token(self, index: int) -> str:
        return self.store.get_token(index)

    def add(self, item: str | Any) -> Vocab:
        if not isinstance(item, str):
            item = item.label
        self.store.add(item)
        return self

    def extend(self, items: Iterable[str | Any]) -> Vocab:
        items = [
            item if isinstance(item, str) else item.label for item in items
        ]
        self.store.extend(items)
        return self

    def items(self, verbose: bool = False) -> Iterable[Tuple[int, str]]:
        return self.store.items(verbose=verbose)

    def __len__(self) -> int:
        return len(self.store)

    def __iter__(self) -> Iterable[str]:
        for _, token in self.items():
            yield token
