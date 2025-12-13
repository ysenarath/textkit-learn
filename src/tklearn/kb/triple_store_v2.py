from __future__ import annotations

from collections import defaultdict

from tklearn.kb.models import Triple

T = dict[str, set[str | tuple[str, int]]]
V = dict[str, dict[int, T]]


class TripleStore:
    def __init__(self):
        self.data: V = {}

    def get(
        self, subject: str | tuple[str, int], default: T = None
    ) -> dict[str, set[tuple[str, int]]] | T:
        """Return all triples in the store."""
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
