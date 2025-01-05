from __future__ import annotations

import itertools
import os
from pathlib import Path
from typing import Any, List, Union

import xxhash
from datasets.utils import _dill as dill

__all__ = [
    "get_content_hash",
]


def get_content_hash(paths: str | Path | List[str | Path]) -> str:
    if isinstance(paths, (str, Path)):
        paths = [paths]
    paths = list(map(str, paths))
    paths = sorted(map(os.path.normpath, paths))
    x_hash = xxhash.xxh32()
    for path in paths:
        # If path is a file, add file content to hash
        if os.path.isfile(path):
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    x_hash.update(chunk)
            continue
        for root, dirs, files in os.walk(path):
            # Sort for consistent ordering
            dirs = sorted(dirs)
            files = sorted(files)
            for file in files:
                filepath = os.path.join(root, file)
                # Add filename to hash
                x_hash.update(filepath.encode())
                # Add file content to hash
                with open(filepath, "rb") as f:
                    for chunk in iter(lambda: f.read(4096), b""):
                        x_hash.update(chunk)
    return x_hash.hexdigest()


def hash_bytes(value: Union[bytes, List[bytes]]) -> str:
    value = [value] if isinstance(value, bytes) else value
    m = xxhash.xxh64()
    for x in value:
        m.update(x)
    return m.hexdigest()


def hash(value: Any, *args: Any) -> str:
    return hash_bytes(map(dill.dumps, itertools.chain([value], args)))
