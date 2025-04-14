import numpy as np
from tqdm import auto as tqdm

from .triple_store import TripleStore
from .wiktionary import Word, parse_jsonl

triplet_path = "wiktionary.triples.duckdb"
input_json_path = "/Users/yasas/Documents/Projects/Experiments/wordex/data/raw-wiktextract-data.jsonl"
total = 9955900
desc = "Extracting senses from wiktionary"

senses: dict[str, set[int]] = {}
gloss2idx: dict[str, int] = {}

for word in tqdm.tqdm(parse_jsonl(input_json_path), total=total, desc=desc):
    if word.lang_code and word.lang_code != "en":
        continue
    for word_sense in word.senses or []:
        if word.word not in senses:
            senses[word.word] = set()
        word_sense_gloss = None
        if word_sense.glosses:
            word_sense_gloss = " ".join(word_sense.glosses or []).strip()
        # DO NOT USE raw_glosses since those are either obsolete words
        #   or not properly defined terms
        if word_sense_gloss is None:
            continue
        gloss_index = gloss2idx.setdefault(word_sense_gloss, len(gloss2idx))
        senses[word.word].add(gloss_index)


def get_embedding(gloss: str) -> np.ndarray:
    raise NotImplementedError


idx2gloss: dict[int, str] = {}
embeddings: dict[int, np.ndarray] = {}

for gloss, index in gloss2idx.items():
    idx2gloss[index] = gloss
    embeddings[index] = get_embedding(gloss)


def closest_sense(word: str, sense_index: int):
    if word not in senses:
        return None
    word_sense_indexes: list[int] = list(senses[word])
    if not word_sense_indexes:
        return
    if sense_index in word_sense_indexes:
        return sense_index
    sense_embeddings = np.array([
        embeddings[index] for index in word_sense_indexes
    ])
    if sense_index is None:
        sense_embedding = np.array([np.mean(sense_embeddings, axis=0)])
    elif not isinstance(sense_index, int):
        raise ValueError("sense_index must be int or None")
    sense_embedding = embeddings[sense_index]
    distances = np.linalg.norm(sense_embeddings - sense_embedding, axis=1)
    return word_sense_indexes[np.argmin(distances)]


predicates = ["synonym", "antonym", "hyponym", "hypernym", "instance"]


def normalize(a, b, c):
    # convert to bottom-up relation if needed
    if b == "hyponym" or b == "instance":
        # here the original relation is top-down
        #   i.e., c is hyponym of a / c is instance of a
        return (c, b, a)
    if b == "hypernym":
        # c is hypernym of a (top-down, i.e., c is at the top)
        b = "hyponym"
        # a is hyponym of c (bottom-up)
        return (a, b, c)
    if b == "synonym" or b == "antonym":
        # symmetrical (not DAG)
        # a is synonym of b == b is synonym of a
        return (a, b, c)
    raise ValueError(f"relation type {b!r} is not recognized")


def get_relations(word: Word):
    for predicate in predicates:
        relations = getattr(word, f"{predicate}s") or []
        for relation in relations:
            if not relation.word:
                continue
            relation_sense = getattr(relation, "sense", None)
            subject = (
                word.word,
                closest_sense(word.word, gloss2idx.get(word.gloss, None)),
            )
            object_ = (relation.word, None)
            yield normalize(subject, predicate, object_)
        for word_sense in word.senses or []:
            relations = getattr(word_sense, f"{predicate}s") or []
            for relation in relations:
                if not relation.word:
                    continue
                sense_gloss = None
                # `relation_sense` this is likely None
                relation_sense = getattr(relation, "sense", None)
                if word_sense.glosses:
                    sense_gloss = " ".join(word_sense.glosses or []).strip()
                elif relation_sense:
                    sense_gloss = relation_sense
                # don't use `word_sense.raw_glosses` because they are not good ones
                subject = (
                    word.word,
                    closest_sense(word.word, gloss2idx.get(sense_gloss, None)),
                )
                object_ = (relation.word, None)
                yield normalize(subject, predicate, object_)


triples = TripleStore(triplet_path, read_only=False)

for word in tqdm.tqdm(parse_jsonl(input_json_path), total=total, desc=desc):
    for relation in get_relations(word):
        subject, predicate, object_ = relation
        subj_word, subj_sense = subject
        if subj_sense is None:
            continue
        triple = ((subj_word, subj_sense), predicate, object_)
        triples.insert(triple)

triples.close()
