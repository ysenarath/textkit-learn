from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator, Mapping, Optional, Union

import lmdb

__all__ = [
    "FileCache",
    "LMDBCache",
]


class BaseCache(Mapping):
    def __getitem__(self, key: str) -> Any:
        raise NotImplementedError

    def __setitem__(self, key: str, value: Any) -> None:
        raise NotImplementedError

    def __delitem__(self, key: str) -> None:
        raise NotImplementedError

    def __iter__(self) -> Generator[str, None, None]:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __contains__(self, key: str) -> bool:
        raise NotImplementedError


class FileCache:
    def __init__(self, path: Union[Path, str]):
        self.path = Path(path)
        if not self.path.exists():
            with open(self.path, "w") as f:
                json.dump({}, f)

    def __getitem__(self, key: str) -> Optional[Any]:
        if not self.path.exists():
            raise KeyError(f"{key}")
        with open(self.path, encoding="utf-8") as f:
            data = json.load(f)
        return data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        with open(self.path, encoding="utf-8") as f:
            data = json.load(f)
        with open(self.path, "w", encoding="utf-8") as f:
            data[key] = value
            json.dump(data, f)

    def __delitem__(self, key: str) -> None:
        with open(self.path, encoding="utf-8") as f:
            data = json.load(f)
        with open(self.path, "w", encoding="utf-8") as f:
            del data[key]
            json.dump(data, f)

    def __contains__(self, key: str) -> bool:
        with open(self.path, encoding="utf-8") as f:
            return key in json.load(f)

    def __iter__(self) -> Generator[str, None, None]:
        with open(self.path, encoding="utf-8") as f:
            yield from json.load(f)

    def __len__(self) -> int:
        with open(self.path, encoding="utf-8") as f:
            return len(json.load(f))


class LMDBCache(BaseCache):
    def __init__(self, path: Union[Path, str], map_size=1024 * 1024 * 1024):
        self.path = Path(path)
        self.map_size = map_size
        self._env: Optional[lmdb.Environment] = None

    @property
    def env(self) -> lmdb.Environment:
        """Lazy initialization of LMDB environment."""
        if self._env is None:
            self._env = lmdb.open(
                str(self.path),
                map_size=self.map_size,
                subdir=True,
                map_async=True,
                writemap=True,
                lock=True,
                max_readers=126,
            )
        return self._env

    @contextmanager
    def begin(
        self, write: bool = False
    ) -> Generator[lmdb.Transaction, None, None]:
        try:
            with self.env.begin(write=write) as txn:
                yield txn
        except lmdb.Error as e:
            # Reset environment on serious errors
            self.close()
            raise e

    def __getitem__(self, key: str) -> Any:
        """Retrieve a value from the cache."""
        encoded_key = key.encode("utf-8")
        with self.begin() as txn:
            value = txn.get(encoded_key)
            if value is None:
                raise KeyError(key)
            return json.loads(value.decode("utf-8"))

    def __setitem__(self, key: str, value: Any) -> None:
        """Store a value in the cache."""
        encoded_key = key.encode("utf-8")
        encoded_value = json.dumps(value).encode("utf-8")
        with self.begin(write=True) as txn:
            txn.put(encoded_key, encoded_value)

    def __contains__(self, key: str) -> bool:
        """Check if a key exists in the cache."""
        encoded_key = key.encode("utf-8")
        with self.begin() as txn:
            return txn.get(encoded_key) is not None

    def close(self) -> None:
        """Close the LMDB environment."""
        if self._env is not None:
            self._env.sync()
            self._env.close()
            self._env = None

    def __enter__(self):
        """Context manager support."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Ensure environment is closed on context exit."""
        self.close()

    def __del__(self):
        """Ensure environment is closed on garbage collection."""
        self.close()
