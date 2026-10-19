from __future__ import annotations

import functools
import os
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import snapshot_download

from tklearn import config
from tklearn.kb.table_store import FORMAT_VERSION, TableStore

__all__ = [
    "WiktionaryStore",
]

TABLES = [
    "glosses.parquet",
    "senses.parquet",
    "lexicon.parquet",
    "triples.parquet",
]


class WiktionaryStore(TableStore):
    """English Wiktionary as a knowledge store.

    Reads the tables from `path`. Those that are missing are downloaded
    from the Hugging Face Hub, once; nothing else contacts the Hub.

    Parameters
    ----------
    version : str, default="v0.5"
        Version of the tables; selects the default `path` and `repo_id`.
    repo_id : str, optional
        Hub dataset to download from; by default
        ``textkit-learn/wiktionary-<version>``.
    path : str or Path, optional
        Directory of the tables; by default
        ``config.assets_dir/wiktionary-<version>``.
    embeddings : bool, default=False
        Also download the gloss embeddings (a few GB). Without them,
        candidates have no embedding unless the file is already in `path`.
    """

    def __init__(
        self,
        version: str = f"v{FORMAT_VERSION}",
        repo_id: str | None = None,
        path: str | Path | None = None,
        embeddings: bool = False,
    ) -> None:
        self._init_kwargs = dict(
            version=version, repo_id=repo_id, path=path, embeddings=embeddings
        )
        self.version = version
        self.repo_id = repo_id or f"textkit-learn/wiktionary-{version}"
        if path is None:
            path = Path(config.assets_dir) / f"wiktionary-{version}"
        path = Path(path)
        # manifest.json last: it marks the tables as complete
        files = TABLES + ["manifest.json"]
        if embeddings:
            files.insert(0, "embeddings.npy")
        missing = [name for name in files if not (path / name).exists()]
        if missing:
            _download(self.repo_id, missing, path)
        super().__init__(path)

    def __reduce__(self):
        return (functools.partial(type(self), **self._init_kwargs), ())

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self._init_kwargs.items())
        return f"{type(self).__name__}({args})"


def _download(repo_id: str, files: list[str], path: Path) -> None:
    """Download files of a Hub dataset into `path`, in order."""
    path.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix=f".{path.name}-", dir=path.parent)
    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=tmp,
            allow_patterns=files,
        )
        for name in files:
            if not (Path(tmp) / name).exists():
                msg = f"{repo_id} has no {name}"
                raise FileNotFoundError(msg)
        for name in files:
            # a file is replaced in one step, so readers never see part of it
            os.replace(Path(tmp) / name, path / name)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
