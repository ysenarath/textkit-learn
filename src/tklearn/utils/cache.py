from __future__ import annotations

import os
import pickle
import time
import uuid
from pathlib import Path
from typing import Any

from tklearn import config
from tklearn.utils import hashing

__all__ = [
    "FileCache",
]


class FileCache:
    def __init__(self, *, temp_dir: str | Path | None = None):
        if temp_dir is None:
            temp_dir = config.cache_dir / "cache"
        self.temp_dir = temp_dir

    def _get_path(self, fingerprint: str) -> Path:
        filename = f"{fingerprint}.pkl"
        base_path = self.temp_dir
        for i in range(3):
            base_path = base_path / fingerprint[i : i + 1]
        return base_path / filename

    def dump(
        self, obj: Any, *, fingerprint: Any = None, exist_ok: bool = False
    ) -> bool:
        if fingerprint is None:
            fingerprint = hashing.hash(obj)

        final_path = self._get_path(fingerprint)

        if final_path.exists():
            if exist_ok:
                return False
            else:
                raise FileExistsError(f"file exists: '{final_path}'")

        uid = uuid.uuid4()
        temp_path = final_path.with_name(f"{final_path.name}.{uid}.tmp")

        # Ensure parent directory exists
        temp_path.parent.mkdir(parents=True, exist_ok=True)

        with temp_path.open("wb") as f:
            pickle.dump(obj, f)

        # Move the file to the cache directory
        try:
            # this may fail if another process
            #   has already created the file
            os.rename(temp_path, final_path)
            return True
        except OSError:
            # Another process beat us to it
            try:
                # Clean up our temp file
                os.unlink(temp_path)
            except OSError:
                pass
            return False

    def load(self, fingerprint: str) -> Any:
        path = self._get_path(fingerprint)
        if path.exists():
            with path.open("rb") as f:
                return pickle.load(f)
        return None

    def exists(self, fingerprint: str) -> bool:
        return self._get_path(fingerprint).exists()

    def get_size(self, fingerprint: str) -> int | None:
        path = self._get_path(fingerprint)
        try:
            return path.stat().st_size if path.exists() else None
        except OSError:
            return None

    def cleanup(self, max_age_hours: int = 24) -> int:
        """Clean up temporary files older than max_age_hours.
        Returns number of files removed."""
        current_time = time.time()
        cleaned = 0

        # Find all .tmp files recursively
        for tmp_file in Path(self.temp_dir).rglob("*.tmp"):
            try:
                stats = tmp_file.stat()
                # Only remove if:
                # 1. File is old enough
                # 2. File size is not changing (indicating active write)
                if current_time - stats.st_mtime > max_age_hours * 3600:
                    # Double check file size hasn't changed
                    time.sleep(0.1)  # Brief pause
                    new_stats = tmp_file.stat()
                    if (
                        new_stats.st_size == stats.st_size
                    ):  # File is not being written to
                        tmp_file.unlink()
                        cleaned += 1
            except OSError:
                continue

        return cleaned
