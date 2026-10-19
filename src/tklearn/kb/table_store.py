from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import unibreak

from tklearn.kb.base import KnowledgeStore, WordSense
from tklearn.kb.models import Triple

__all__ = [
    "FORMAT_VERSION",
    "TableStore",
    "form_key",
    "split_words",
    "write_tables",
]

#: Version of the table layout, the tklearn version that introduced it.
FORMAT_VERSION = "0.5"

GLOSSES = pa.schema([("sense_id", pa.int64()), ("gloss", pa.string())])
SENSES = pa.schema([("word", pa.string()), ("sense_id", pa.int64())])
LEXICON = pa.schema([
    ("form", pa.string()),
    ("word", pa.string()),
    ("sense_id", pa.int64()),
])
TRIPLES = pa.schema([
    ("subject", pa.string()),
    ("subject_sense", pa.int64()),
    ("predicate", pa.string()),
    ("object", pa.string()),
    ("object_sense", pa.int64()),
])


def split_words(text: str) -> list[tuple[str, int, int]]:
    """Split a text into lowercased words with their character offsets."""
    words = []
    end = 0
    for token in unibreak.split_words(text):
        start, end = end, end + len(token)
        if not token.isspace():
            words.append((token.lower(), start, end))
    return words


def form_key(form: str) -> str:
    """Return the key a form is matched by: its words joined by spaces."""
    return " ".join(word for word, _, _ in split_words(form))


class TableStore(KnowledgeStore):
    """A knowledge store read from Parquet tables written by `write_tables`.

    Parameters
    ----------
    path : str or Path
        Directory holding the tables.

    Attributes
    ----------
    path : Path
        Directory holding the tables.
    metadata : dict
        What the tables were built from, as given to `write_tables`.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        with open(self.path / "manifest.json") as f:
            manifest = json.load(f)
        if manifest.get("format") != FORMAT_VERSION:
            msg = (
                f"{self.path} has table format {manifest.get('format')!r}, "
                f"expected {FORMAT_VERSION!r}"
            )
            raise ValueError(msg)
        self.metadata = manifest.get("metadata", {})
        glosses = _read(self.path / "glosses.parquet")
        self._glosses = glosses["gloss"].combine_chunks()
        self._senses = _read(self.path / "senses.parquet")
        self._sense_rows = _runs(self._senses["word"])
        self._lexicon = _read(self.path / "lexicon.parquet")
        self._form_rows = _runs(self._lexicon["form"])
        # longest form in words, by its first word
        self._form_len: dict[str, int] = {}
        for key in self._form_rows:
            first, _, rest = key.partition(" ")
            n = rest.count(" ") + 2 if rest else 1
            if n > self._form_len.get(first, 0):
                self._form_len[first] = n
        self._triples = _read(self.path / "triples.parquet")
        self._subject_rows = _runs(self._triples["subject"])
        self._by_object: tuple[np.ndarray, dict] | None = None
        self._embeddings: np.ndarray | None = None

    def __reduce__(self):
        return (type(self), (self.path,))

    def __repr__(self) -> str:
        return f"{type(self).__name__}({str(self.path)!r})"

    def find_forms(
        self, text: str
    ) -> Iterator[tuple[int, int, list[WordSense]]]:
        words = split_words(text)
        for i, (first, start, _) in enumerate(words):
            key = ""
            for word, _, end in words[i : i + self._form_len.get(first, 0)]:
                key = f"{key} {word}" if key else word
                rows = self._form_rows.get(key)
                if rows is None:
                    continue
                matches = self._lexicon.slice(rows[0], rows[1] - rows[0])
                yield (
                    start,
                    end,
                    list(
                        zip(
                            matches["word"].to_pylist(),
                            matches["sense_id"].to_pylist(),
                        )
                    ),
                )

    def get_gloss(self, sense_id: int) -> str:
        if not 0 <= sense_id < len(self._glosses):
            raise KeyError(sense_id)
        return self._glosses[sense_id].as_py()

    def get_senses(self, word: str) -> list[int]:
        rows = self._sense_rows.get(word)
        if rows is None:
            return []
        senses = self._senses["sense_id"].slice(rows[0], rows[1] - rows[0])
        return senses.to_pylist()

    def get_relations(
        self,
        word: str,
        sense_id: int | None = None,
        predicates: Iterable[str] | None = None,
        by: str = "subject",
    ) -> list[Triple]:
        if by == "subject":
            rows = self._subject_rows.get(word)
            if rows is None:
                return []
            table = self._triples.slice(rows[0], rows[1] - rows[0])
        elif by == "object":
            order, object_rows = self._object_index()
            rows = object_rows.get(word)
            if rows is None:
                return []
            table = self._triples.take(order[rows[0] : rows[1]])
        else:
            msg = f"by must be 'subject' or 'object', got {by!r}"
            raise ValueError(msg)
        # filter before converting rows to Python, which is the slow part
        mask = None
        if sense_id is not None:
            senses = table[f"{by}_sense"]
            # a triple with no sense holds for every sense
            mask = pc.or_(
                pc.is_null(senses),
                pc.fill_null(pc.equal(senses, sense_id), False),
            )
        if predicates is not None:
            value_set = pa.array(list(predicates), pa.string())
            keep = pc.is_in(table["predicate"], value_set=value_set)
            mask = keep if mask is None else pc.and_(mask, keep)
        if mask is not None:
            table = table.filter(mask)
        return [
            Triple((s, ss), p, (o, os_))
            for s, ss, p, o, os_ in zip(*table.to_pydict().values())
        ]

    def get_embedding(self, sense_id: int) -> np.ndarray | None:
        if self._embeddings is None:
            path = self.path / "embeddings.npy"
            if not path.exists():
                return None
            self._embeddings = np.load(path, mmap_mode="r")
        return np.array(self._embeddings[sense_id])

    def _object_index(self) -> tuple[np.ndarray, dict]:
        # built on first use: the triples are stored sorted by subject
        if self._by_object is None:
            order = pc.sort_indices(self._triples["object"])
            objects = self._triples["object"].take(order)
            self._by_object = (order.to_numpy(), _runs(objects))
        return self._by_object


def write_tables(
    path: str | Path,
    glosses: Sequence[str],
    senses: Iterable[tuple[str, int]],
    lexicon: Iterable[tuple[str, str, int | None]],
    triples: Iterable[tuple[str, int | None, str, str, int | None]],
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """Write the tables of a `TableStore` to a directory.

    The tables are written to a temporary directory next to `path`, which
    then replaces `path`, so a reader never sees a partial store.

    Parameters
    ----------
    path : str or Path
        Directory to write.
    glosses : sequence of str
        The gloss of each sense; its position is the sense id.
    senses : iterable of (word, sense_id)
        The senses of each word.
    lexicon : iterable of (form, word, sense_id)
        Forms that refer to a word, or to one sense of it.
    triples : iterable of (subject, subject_sense, predicate, object, \
object_sense)
        Relations between words; a sense of None means the word in general.
    metadata : mapping, optional
        What the tables were built from, saved in ``manifest.json``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lexicon = (
        (key, word, sense)
        for form, word, sense in lexicon
        if (key := form_key(form))
    )
    tables = {
        "glosses": pa.table(
            [pa.array(np.arange(len(glosses))), pa.array(glosses)],
            schema=GLOSSES,
        ),
        "senses": _table(senses, SENSES),
        "lexicon": _table(lexicon, LEXICON),
        "triples": _table(triples, TRIPLES),
    }
    manifest = {"format": FORMAT_VERSION, "metadata": dict(metadata or {})}
    tmp = Path(tempfile.mkdtemp(prefix=f".{path.name}-", dir=path.parent))
    # mkdtemp makes the directory private; give it the usual permissions so
    # that others sharing the directory (e.g. on NFS) can read the store
    umask = os.umask(0)
    os.umask(umask)
    os.chmod(tmp, 0o777 & ~umask)
    try:
        for name, table in tables.items():
            pq.write_table(table, tmp / f"{name}.parquet")
        with open(tmp / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        if path.exists():
            old = Path(
                tempfile.mkdtemp(prefix=f".{path.name}-", dir=path.parent)
            )
            os.replace(path, old / path.name)
            os.replace(tmp, path)
            shutil.rmtree(old)
        else:
            os.replace(tmp, path)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _table(rows: Iterable[tuple], schema: pa.Schema) -> pa.Table:
    """Make a table of unique rows, sorted by every column in order."""
    columns = list(zip(*rows)) or [()] * len(schema)
    table = pa.table(
        [pa.array(c, type=f.type) for c, f in zip(columns, schema)],
        schema=schema,
    )
    names = schema.names
    table = table.group_by(names, use_threads=False).aggregate([])
    return table.select(names).sort_by([(n, "ascending") for n in names])


def _read(path: Path) -> pa.Table:
    # one chunk per column: taking rows from many chunks is slow
    return pq.read_table(path, memory_map=True).combine_chunks()


def _runs(column: pa.ChunkedArray | pa.Array) -> dict[str, tuple[int, int]]:
    """Map each value of a sorted column to its ``(start, stop)`` rows."""
    if isinstance(column, pa.ChunkedArray):
        column = column.combine_chunks()
    if len(column) == 0:
        return {}
    encoded = column.dictionary_encode()
    codes = encoded.indices.to_numpy(zero_copy_only=False)
    starts = np.flatnonzero(np.r_[True, codes[1:] != codes[:-1]])
    stops = np.r_[starts[1:], len(codes)]
    keys = encoded.dictionary.take(pa.array(codes[starts])).to_pylist()
    return dict(zip(keys, zip(starts.tolist(), stops.tolist())))
