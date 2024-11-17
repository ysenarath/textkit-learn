from __future__ import annotations

import threading
from dataclasses import MISSING
from typing import Dict, Generator, NamedTuple, Tuple

from sqlalchemy import (
    Column,
    Index,
    Integer,
    String,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, scoped_session, sessionmaker
from sqlalchemy.pool import QueuePool


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


class SQLAlchemyVocabStore:
    """SQLAlchemy-based persistent vocabulary implementation with advanced features."""

    def __init__(
        self,
        url: str,
        cache_size: int = 10000,
        verbose: bool = False,
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

    def items(self) -> Generator[Tuple[VocabItem], None, None]:
        """Iterate over all tokens in the vocabulary."""
        session = self.Session()
        try:
            for vocab_token in session.query(VocabToken).all():
                yield VocabItem(vocab_token.id, vocab_token.token)
        finally:
            session.close()


def example_usage():
    # remove tmp/vocab.db if it exists
    import os

    if os.path.exists("tmp/vocab.db"):
        os.remove("tmp/vocab.db")

    # Initialize vocabulary with SQLite database
    vocab = SQLAlchemyVocabStore("sqlite:///tmp/vocab.db", verbose=False)

    # Add tokens to the vocabulary
    vocab.add("hello", 101)
    vocab.add("world")

    # Check for the token "hello"
    token_id = vocab.get_index("hello")
    print(f"Token ID for 'hello': {token_id}")
    token = vocab.get_token(101)
    print(f"Token for ID 101: {token}")

    # Check for the token "world"
    token_id = vocab.get_index("world")
    print(f"Token ID for 'world': {token_id}")  # 102 (probably)

    # Check for a non-existent token
    try:
        token_id = vocab.get_index("non-existent")
    except KeyError as e:
        print(e)

    # Check for a non-existent index
    try:
        token = vocab.get_token(999)
    except IndexError as e:
        print(e)

    # Check vocabulary size
    print(f"Vocabulary size: {len(vocab)}")


if __name__ == "__main__":
    example_usage()
