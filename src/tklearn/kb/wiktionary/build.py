from __future__ import annotations

import datetime
import gzip
import os
import re
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import orjson
from tqdm.auto import tqdm

from tklearn import __version__, config
from tklearn.kb.table_store import FORMAT_VERSION, write_tables

__all__ = [
    "DUMP_URL",
    "PREDICATES",
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
    if path is None:
        path = Path(config.assets_dir) / f"wiktionary-v{FORMAT_VERSION}"
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
