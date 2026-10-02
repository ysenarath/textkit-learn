from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "Config",
    "config",
]

DEFAULT_CACHE_PATH = Path.home() / ".cache" / "tklearn"
DEFAULT_DATASET_BATCH_SIZE = 1000


def _default_base_dir() -> Path:
    return Path(os.getenv("TKLEARN_CACHE", DEFAULT_CACHE_PATH)).absolute()


def is_debug_enabled() -> bool:
    value = str(os.getenv("TKLEARN_DEBUG", "0")).lower()
    return value in ("1", "true", "yes", "on")


@dataclass
class Config:
    """Global settings. Change them on `tklearn.config`.

    Attributes
    ----------
    base_dir : Path
        Root of all tklearn files; ``$TKLEARN_CACHE`` or ``~/.cache/tklearn``.
    dataset_batch_size : int
        Default batch size for dataset mapping utilities.
    debug : bool
        Debug logging; enabled by ``TKLEARN_DEBUG=1``.
    """

    base_dir: Path = field(default_factory=_default_base_dir)
    dataset_batch_size: int = DEFAULT_DATASET_BATCH_SIZE
    debug: bool = field(default_factory=is_debug_enabled)

    @property
    def cache_dir(self) -> Path:
        """Cache files (see `tklearn.utils.cache`)."""
        return Path(self.base_dir) / "cache"

    @property
    def temp_dir(self) -> Path:
        """Temporary files."""
        return Path(self.base_dir) / "temp"

    @property
    def assets_dir(self) -> Path:
        """Downloaded and generated resources (embeddings, knowledge bases)."""
        return Path(self.base_dir) / "assets"


config = Config()
