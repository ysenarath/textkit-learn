from pathlib import Path
from typing import Any, Generator, Tuple

import duckdb
import networkx as nx

from tklearn.kb.models import Triple

CREATE_TABLE_EXPR = """CREATE TABLE IF NOT EXISTS triples (
    subject_word TEXT NOT NULL,
    subject_sense INTEGER NOT NULL,
    predicate TEXT NOT NULL,
    object_word TEXT NOT NULL,
    object_sense INTEGER NOT NULL
)"""

CREATE_INDEX_EXPR = """CREATE UNIQUE INDEX IF NOT EXISTS uk_triples ON triples (
    subject_word, subject_sense, predicate, object_word, object_sense
)"""


class TripleStore:
    def __init__(self, path: str | Path, read_only: bool = False):
        self.path = path
        self.con = duckdb.connect(path, read_only=read_only)
        if read_only:
            return
        self.create_table()

    def close(self):
        self.con.close()

    def create_table(self):
        self.con.execute(CREATE_TABLE_EXPR)
        self.con.execute(CREATE_INDEX_EXPR)

    def insert(self, triple: Triple, *args: Any):
        if len(args) > 0:
            triple = (triple, *args)
        self.con.execute(
            "INSERT OR IGNORE INTO triples VALUES (?, ?, ?, ?, ?)",
            Triple.to_tuple(triple),
        )

    @staticmethod
    def _add_filter_query(
        q: str, params: list, triple: Triple
    ) -> Tuple[str, list]:
        triple = Triple.to_tuple(triple)
        subject_word, subject_sense, predicate, object_word, object_sense = (
            triple
        )
        q += " OR (1=1"
        if subject_word is not None:
            q += " AND subject_word COLLATE NOCASE.NOACCENT = ?"
            params.append(subject_word)
        if subject_sense >= 0:
            q += " AND (subject_sense = ? OR subject_sense < 0)"
            params.append(subject_sense)
        if predicate is not None:
            q += " AND predicate COLLATE NOCASE.NOACCENT = ?"
            params.append(predicate)
        if object_word is not None:
            q += " AND object_word COLLATE NOCASE.NOACCENT = ?"
            params.append(object_word)
        if object_sense >= 0:
            q += " AND (object_sense = ? OR object_sense < 0)"
            params.append(object_sense)
        q += ")"
        return q, params

    def query(self, *triples: Triple | tuple) -> Generator[Triple, None, None]:
        # 1=0 always false, so we can start with OR
        q = "SELECT * FROM triples"
        params = []

        if triples:
            q += " WHERE 1=0"

        for triple in triples:
            q, params = self._add_filter_query(q, params, triple)

        for row in self.con.execute(q, params).fetchall():
            yield Triple.from_tuple(row)

    def __len__(self):
        return self.con.execute("SELECT COUNT(*) FROM triples").fetchone()[0]

    def __iter__(self):
        return self.query()

    def to_networkx(self):
        G = nx.Graph()
        for triple in self:
            subject, predicate, object = triple
            G.add_edge(subject, object, predicate=predicate)
        return G
