from __future__ import annotations

import datetime
import gzip
import json
import os
import re
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import orjson
import pyarrow.parquet as pq
from tqdm.auto import tqdm

from tklearn import __version__, config
from tklearn.embeddings import Embedding
from tklearn.kb.table_store import FORMAT_VERSION, write_tables

__all__ = [
    "DUMP_URL",
    "PREDICATES",
    "build_embeddings",
    "build_wiktionary",
]

#: The kaikki.org dump of all Wiktionary entries that the tables are built
#: from.
DUMP_URL = "https://kaikki.org/dictionary/raw-wiktextract-data.jsonl.gz"
#: Relations taken from the dump.
PREDICATES = ("synonym", "antonym", "hypernym", "hyponym", "category")

# relation -> the field of an entry or sense that lists it
_FIELDS = {
    "synonym": "synonyms",
    "antonym": "antonyms",
    "hypernym": "hypernyms",
    "hyponym": "hyponyms",
    "category": "categories",
}
_ENGLISH = re.compile(rb'"lang_code":\s*"en"')
_ANTI_PATTERNS = [
    # antonym(s) of "{definition}" -> definition
    # (see https://en.wiktionary.org/wiki/Template:antsense)
    re.compile(r'antonym\(s\) of [“"](.*)[”"]')
]


def build_wiktionary(
    dump: str | Path,
    path: str | Path | None = None,
    predicates: Iterable[str] = PREDICATES,
) -> None:
    """Build the tables of a `WiktionaryStore` from a kaikki.org dump.

    Download the dump from `DUMP_URL` first (about 3 GB). Only English
    entries are read.

    Each gloss is a sense, shared by the words that have it. A relation
    with a label of which sense it is about, but no gloss, gets that label
    as its sense. Hyponyms are stored as the hypernyms they imply, and
    relations of a word to itself are skipped: they come from Thesaurus
    lists that Wiktionary copies into the entry of each word they list.

    Parameters
    ----------
    dump : str or Path
        The dump, ``.jsonl`` or ``.jsonl.gz``.
    path : str or Path, optional
        Directory to write; by default the one `WiktionaryStore()` reads,
        ``config.assets_dir/wiktionary-v0.5``.
    predicates : iterable of str, default=PREDICATES
        Relations to keep.
    """
    dump = Path(dump)
    path = _default_path(path)
    predicates = sorted(set(predicates))
    unknown = set(predicates) - set(PREDICATES)
    if unknown:
        msg = f"unknown predicates {sorted(unknown)}, expected {PREDICATES}"
        raise ValueError(msg)
    builder = _Builder(predicates)
    with (
        open(dump, "rb") as raw,
        tqdm(
            total=os.path.getsize(dump),
            unit="B",
            unit_scale=True,
            desc="Reading Wiktionary",
        ) as progress,
    ):
        lines = gzip.GzipFile(fileobj=raw) if dump.suffix == ".gz" else raw
        for i, line in enumerate(lines):
            if i % 10_000 == 0:
                progress.update(raw.tell() - progress.n)
            # most entries are in other languages; skip them unparsed
            if not _ENGLISH.search(line):
                continue
            entry = orjson.loads(line)
            if entry.get("lang_code") == "en" and _is_str(entry.get("word")):
                builder.add_entry(entry)
        progress.update(raw.tell() - progress.n)
    modified = datetime.date.fromtimestamp(dump.stat().st_mtime)
    write_tables(
        path,
        glosses=list(builder.sense_ids),
        senses=builder.senses,
        lexicon=builder.lexicon,
        triples=builder.triples,
        metadata={
            "source": DUMP_URL,
            "dump": dump.name,
            "dump_modified": modified.isoformat(),
            "language": "en",
            "predicates": predicates,
            "built_with": f"tklearn {__version__}",
        },
    )


def build_embeddings(
    embedding: Embedding,
    path: str | Path | None = None,
    batch_size: int = 10_000,
) -> None:
    """Embed the glosses of a store and save them as its ``embeddings.npy``.

    Progress is saved after each batch: run it again with the same model to
    continue after an interruption. The model is recorded in the store's
    metadata, under ``"embeddings"``.

    Parameters
    ----------
    embedding : Embedding
        The model, e.g. ``OpenRouterEmbedding("qwen/qwen3-embedding-8b",
        dimensions=1024)``; glosses are embedded with
        `Embedding.encode_document`.
    path : str or Path, optional
        Directory of the store's tables; by default the one
        `WiktionaryStore()` reads, ``config.assets_dir/wiktionary-v0.5``.
    batch_size : int, default=10_000
        Glosses embedded between saves.
    """
    path = _default_path(path)
    glosses = pq.read_table(path / "glosses.parquet")["gloss"]
    if len(glosses) == 0:
        msg = f"{path} has no glosses to embed"
        raise ValueError(msg)
    # the repr tells apart settings that change the vectors, e.g. their size
    model = {"model": embedding.name, "embedding": repr(embedding)}
    partial = path / "embeddings.npy.partial"
    progress_file = path / "embeddings.progress.json"
    done, vectors = 0, None
    if partial.exists() and progress_file.exists():
        progress = json.loads(progress_file.read_text())
        if progress["model"] == model:
            done = progress["done"]
            vectors = np.lib.format.open_memmap(partial, mode="r+")
    with tqdm(
        total=len(glosses), initial=done, desc="Embedding glosses"
    ) as bar:
        while done < len(glosses):
            batch = glosses.slice(done, batch_size).to_pylist()
            embedded = embedding.encode_document(batch)
            if vectors is None:
                vectors = np.lib.format.open_memmap(
                    partial,
                    mode="w+",
                    dtype=np.float32,
                    shape=(len(glosses), embedded.shape[1]),
                )
            vectors[done : done + len(batch)] = embedded
            vectors.flush()
            done += len(batch)
            _write_json(progress_file, {"model": model, "done": done})
            bar.update(len(batch))
    dim = vectors.shape[1]
    del vectors
    os.replace(partial, path / "embeddings.npy")
    manifest = json.loads((path / "manifest.json").read_text())
    manifest["metadata"]["embeddings"] = {
        **model,
        "dim": dim,
        "built": datetime.date.today().isoformat(),
    }
    _write_json(path / "manifest.json", manifest)
    progress_file.unlink()


def _default_path(path: str | Path | None) -> Path:
    if path is None:
        return Path(config.assets_dir) / f"wiktionary-v{FORMAT_VERSION}"
    return Path(path)


def _write_json(path: Path, data: Any) -> None:
    # replaced in one step, so a reader never sees a partial file
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, path)


class _Builder:
    def __init__(self, predicates: list[str]) -> None:
        self.predicates = predicates
        # gloss -> sense id, in the order glosses are first seen
        self.sense_ids: dict[str, int] = {}
        self.senses: set[tuple[str, int]] = set()
        self.lexicon: set[tuple[str, str, int | None]] = set()
        self.triples: list[tuple] = []

    def get_sense_id(self, gloss: str) -> int:
        for pattern in _ANTI_PATTERNS:
            match = pattern.match(gloss)
            if match:
                gloss = match.group(1)
                break
        return self.sense_ids.setdefault(gloss, len(self.sense_ids))

    def add_entry(self, entry: dict[str, Any]) -> None:
        word = sys.intern(entry["word"])
        self.lexicon.add((word, word, None))
        for form in entry.get("forms") or []:
            if _is_str(form.get("form")):
                self.lexicon.add((form["form"], word, None))
        self.add_relations(word, None, entry)
        for sense in entry.get("senses") or []:
            gloss = " ".join(sense.get("glosses") or []).strip()
            if not gloss:
                gloss = " ".join(sense.get("raw_glosses") or []).strip()
            sense_id = None
            if gloss:
                sense_id = self.get_sense_id(gloss)
                self.senses.add((word, sense_id))
            # an inflection refers to its lemma, with its own gloss
            # (e.g. "plural of dog")
            for lemma in sense.get("form_of") or []:
                if _is_str(lemma.get("word")):
                    self.lexicon.add((word, lemma["word"], sense_id))
            self.add_relations(word, sense_id, sense)

    def add_relations(
        self, word: str, sense_id: int | None, item: dict[str, Any]
    ) -> None:
        for predicate in self.predicates:
            for related in item.get(_FIELDS[predicate]) or []:
                subject_sense = sense_id
                if predicate == "category":
                    obj = related
                else:
                    obj = related.get("word")
                    label = related.get("sense")
                    if subject_sense is None and _is_str(label) and label:
                        # the label says which sense of the word it is about
                        subject_sense = self.get_sense_id(label)
                        self.senses.add((word, subject_sense))
                if not _is_str(obj) or obj == word:
                    continue
                obj = sys.intern(obj)
                if predicate == "hyponym":
                    triple = (obj, None, "hypernym", word, subject_sense)
                else:
                    triple = (word, subject_sense, predicate, obj, None)
                self.triples.append(triple)


def _is_str(value: Any) -> bool:
    return isinstance(value, str)
