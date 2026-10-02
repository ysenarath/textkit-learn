from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Generator


@contextmanager
def change_dir(path: str | Path) -> Generator[None]:
    """Temporarily run in `path` (created if needed).

    gensim and fastText download into the working directory.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    old_path = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old_path)
