from __future__ import annotations

import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any

from tklearn.kb.models import Triple

__all__ = [
    "TripleStore",
    "load_pickle",
]

T = dict[str, set[str | tuple[str, int]]]
V = dict[str, dict[int, T]]

# pickles written before 0.5 refer to the store by its old module path
_RENAMED_MODULES = {"tklearn.kb.triple_store_v2": "tklearn.kb.triple_store"}


class _Unpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        return super().find_class(_RENAMED_MODULES.get(module, module), name)


def load_pickle(path: str | Path) -> Any:
    """Unpickle a file, accepting module paths used by older versions."""
    with open(path, "rb") as f:
        return _Unpickler(f).load()


class TripleStore:
    """In-memory index of triples by subject word and sense.

    ``data[word][sense_id][predicate]`` is the set of ``(word, sense_id)``
    objects. A sense id of None means the relation holds for the word in
    general.
    """

    def __init__(self):
        self.data: V = {}

    def get(
        self, subject: str | tuple[str, int], default: T = None
    ) -> dict[str, set[tuple[str, int]]] | T:
        """Return ``{predicate: objects}`` for a subject.

        A bare word, or a ``(word, None)`` subject, merges the relations of
        all its senses. Returns `default` when the subject is unknown.
        """
        if isinstance(subject, str):
            subj_word, subj_sense = subject, None
        else:
            subj_word, subj_sense = subject
        senses = self.data.get(subj_word)
        if senses is None:
            return default
        if subj_sense is None:
            relations = defaultdict(set)
            for sense_dict in senses.values():
                for predicate, objects in sense_dict.items():
                    relations[predicate].update(objects)
        elif subj_sense in senses:
            relations = defaultdict(set)
            for predicate, objects in senses[subj_sense].items():
                relations[predicate].update(objects)
        else:
            return default
        return dict(relations)

    def add(self, triple: Triple | tuple):
        """Append a triple to the store."""
        if not isinstance(triple, Triple):
            triple = Triple.from_tuple(triple)
        sub, pred, obj = triple
        subj_word, subj_sense_id = sub
        # Initialize nested dictionaries and sets as needed
        if subj_word not in self.data:
            self.data[subj_word] = {}
        if subj_sense_id not in self.data[subj_word]:
            self.data[subj_word][subj_sense_id] = {}
        if pred not in self.data[subj_word][subj_sense_id]:
            self.data[subj_word][subj_sense_id][pred] = set()
        self.data[subj_word][subj_sense_id][pred].add(obj)
