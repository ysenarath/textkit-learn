import hashlib
from os import walk
from os.path import (
    basename,
    dirname,
    isdir,
    isfile,
    normpath,
)
from os.path import exists as path_exists
from os.path import join as path_join
from pathlib import Path
from typing import List, Protocol


class Hash(Protocol):
    def update(self, buf: bytes) -> None: ...


def _update_checksum(checksum: Hash, dirname: str, filenames: str) -> None:
    for filename in sorted(filenames):
        path = path_join(dirname, filename)
        if isfile(path):
            fh = open(path, "rb")
            while 1:
                buf = fh.read(4096)
                if not buf:
                    break
                checksum.update(buf)
            fh.close()


def checksum(paths: str | Path | List[str | Path]) -> str:
    if isinstance(paths, (str, Path)):
        paths = [paths]
    paths = [str(p) for p in paths]
    if not hasattr(paths, "__iter__"):
        raise TypeError("sequence or iterable expected not %r!" % type(paths))
    chksum = hashlib.sha1()
    for path in sorted([normpath(f) for f in paths]):
        if not path_exists(path):
            continue
        if isdir(path):
            walk(path, _update_checksum, chksum)
        elif isfile(path):
            _update_checksum(chksum, dirname(path), basename(path))
    return chksum.hexdigest()
