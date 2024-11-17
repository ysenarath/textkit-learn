from __future__ import annotations

from typing import (
    Any,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
)

import plyvel

__all__ = [
    "TripletStore",
]

# A triplet is represented as a tuple of (subject, predicate, object)
Triplet = Tuple[str, str, str]


class TripletStore:
    """A LevelDB-backed triplet store for storing subject-predicate-object relationships."""

    def __init__(self, path: str):
        """Initialize the triplet store with a path to the LevelDB database."""
        self.path = path
        self.open()

    def open(self):
        """Open the database connection."""
        self.db = plyvel.DB(self.path, create_if_missing=True)
        # Create separate databases for different index types
        # Subject-Predicate-Object index
        self.spo_db = self.db.prefixed_db(b"spo:")
        # Predicate-Object-Subject index
        self.pos_db = self.db.prefixed_db(b"pos:")
        # Object-Subject-Predicate index
        self.osp_db = self.db.prefixed_db(b"osp:")
        return self

    @property
    def frozen(self) -> bool:
        """Check if the database is frozen."""
        return self.db.get(b"__frozen__") is not None

    @frozen.setter
    def frozen(self, __value: bool):
        if __value:
            self.db.put(b"__frozen__", b"")
        else:
            self.db.delete(b"__frozen__")

    def close(self):
        """Close the database connection."""
        self.db.close()
        self.db = None
        self.spo_db = None
        self.pos_db = None
        self.osp_db = None

    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _make_key(self, *parts: str) -> bytes:
        """Create a composite key from parts."""
        return b"\x00".join(part.encode("utf-8") for part in parts)

    def _add(self, triplet: Triplet, **kwargs: dict[str, Any]) -> None:
        subject, predicate, object_ = triplet
        # Store the triplet in all three indices for efficient querying
        spo_key = self._make_key(subject, predicate, object_)
        pos_key = self._make_key(predicate, object_, subject)
        osp_key = self._make_key(object_, subject, predicate)
        if kwargs:
            # Use empty string as value since the data is in the key
            kwargs["spo"].put(spo_key, b"")
            kwargs["pos"].put(pos_key, b"")
            kwargs["osp"].put(osp_key, b"")
        else:
            self.spo_db.put(spo_key, b"")
            self.pos_db.put(pos_key, b"")
            self.osp_db.put(osp_key, b"")

    def add(
        self, triplet: Triplet | Iterable[Triplet], batch_size: int = 1000
    ) -> None:
        """Add a triplet to the store."""
        if isinstance(triplet, tuple):
            self._add(triplet)
            return
        kwargs = None
        for i, t in enumerate(triplet):
            if kwargs is None:
                kwargs = {
                    "spo": self.spo_db.write_batch(),
                    "pos": self.pos_db.write_batch(),
                    "osp": self.osp_db.write_batch(),
                }
            self._add(t, **kwargs)
            if i % batch_size == 0:
                for writer in kwargs.values():
                    writer.write()
                kwargs = None
        if kwargs is not None:
            for writer in kwargs.values():
                writer.write()

    def remove(self, triplet: Triplet) -> None:
        """Remove a triplet from the store."""
        subject, predicate, object_ = triplet
        # key extraction
        spo_key = self._make_key(subject, predicate, object_)
        pos_key = self._make_key(predicate, object_, subject)
        osp_key = self._make_key(object_, subject, predicate)
        # delete
        self.spo_db.delete(spo_key)
        self.pos_db.delete(pos_key)
        self.osp_db.delete(osp_key)

    def _iter_prefix(self, db, prefix: bytes) -> Iterator[Tuple[bytes, bytes]]:
        """Iterate over all entries in the database with the given prefix."""
        return db.iterator(prefix=prefix)

    def find_by_subject(self, subject: str) -> List[Triplet]:
        """Find all triplets with the given subject."""
        prefix = subject.encode("utf-8")
        results = []

        for key, _ in self._iter_prefix(self.spo_db, prefix):
            parts = [part.decode("utf-8") for part in key.split(b"\x00")]
            if len(parts) == 3:
                results.append((
                    parts[0],
                    parts[1],
                    parts[2],
                ))  # Create tuple directly

        return results

    def find_by_predicate(self, predicate: str) -> List[Triplet]:
        """Find all triplets with the given predicate."""
        prefix = predicate.encode("utf-8")
        results = []

        for key, _ in self._iter_prefix(self.pos_db, prefix):
            parts = [part.decode("utf-8") for part in key.split(b"\x00")]
            if len(parts) == 3:
                results.append((
                    parts[2],
                    parts[0],
                    parts[1],
                ))  # Create tuple directly

        return results

    def find_by_object(self, object: str) -> List[Triplet]:
        """Find all triplets with the given object."""
        prefix = object.encode("utf-8")
        results = []

        for key, _ in self._iter_prefix(self.osp_db, prefix):
            parts = [part.decode("utf-8") for part in key.split(b"\x00")]
            if len(parts) == 3:
                results.append((
                    parts[1],
                    parts[2],
                    parts[0],
                ))  # Create tuple directly

        return results

    def find(
        self,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        object: Optional[str] = None,
    ) -> List[Triplet]:
        """
        Find triplets matching the given pattern. Any component can be None to match any value.
        Returns the most efficient query based on which components are specified.
        """
        if subject is not None:
            results = self.find_by_subject(subject)
            if predicate is not None:
                results = [
                    t for t in results if t[1] == predicate
                ]  # Access tuple index 1 for predicate
            if object is not None:
                results = [
                    t for t in results if t[2] == object
                ]  # Access tuple index 2 for object
            return results

        if predicate is not None:
            results = self.find_by_predicate(predicate)
            if object is not None:
                results = [
                    t for t in results if t[2] == object
                ]  # Access tuple index 2 for object
            return results

        if object is not None:
            return self.find_by_object(object)

        # If no components specified, return all triplets
        return self.find_by_subject("")
