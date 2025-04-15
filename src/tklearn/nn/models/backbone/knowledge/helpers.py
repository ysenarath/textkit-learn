from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Tuple, TypeVar

import nltk
import numpy as np
import unibreak
from nltk.corpus import stopwords as st
from scipy.spatial.distance import cdist
from tqdm import auto as tqdm

from tklearn.embeddings import Embedding
from tklearn.kb.lexicon import Lexicon
from tklearn.utils.lexrank import degree_centrality_scores

T = TypeVar("T")

nltk.download("stopwords", quiet=True)

stopwords = set(st.words("english"))

DEFAULT_TOP_K = 3


def preprocess(s: str) -> Optional[str]:
    s = " ".join(s.split()).lower()
    if s in stopwords:
        return None
    return s


def score_triples(
    triples: Dict[Tuple[str, str, str], set], embedding: Embedding
) -> Dict[Tuple[str, str, str], float]:
    vocab = set()
    for s, p, o in triples.keys():
        vocab.update([s, o])
    vocab = list(vocab)
    ndim = embedding.shape[1]
    vectors = np.zeros((len(vocab), ndim))
    for i, term in enumerate(vocab):
        vectors[i] = embedding.get_embedding(term)
    similarity_matrix = 1 - cdist(vectors, vectors, metric="cosine")
    try:
        scores = degree_centrality_scores(similarity_matrix, threshold=0.1)
    except ValueError:
        scores = np.zeros(len(vocab))
    scores = dict(zip(vocab, scores))
    triple_score = {}
    for s, p, o in triples.keys():
        avg_score = (scores[s] + scores[o]) / 2
        triple_score[(s, p, o)] = avg_score
    return triple_score


def score_triples_v2(
    triples: Dict[Tuple[str, str, str], set],
    embedding: Embedding,
    threshold: float = 0.1,
) -> Dict[Tuple[str, str, str], float]:
    vocab2triple = {}
    for s, p, o in triples.keys():
        if p == "HasContext":
            key = f"{s} has context {o}"
        elif p == "IsA":
            key = f"{s} is a {o}"
        else:
            raise ValueError(f"predicate {p} is not supported")
        if key in vocab2triple:
            raise ValueError(f"key {key} is already in the vocab")
        vocab2triple[key] = (s, p, o)
    vocab = list(vocab2triple.keys())
    ndim = embedding.shape[1]
    vectors = np.zeros((len(vocab), ndim))
    for i, term in enumerate(vocab):
        vectors[i] = embedding.get_embedding(term)
    similarity_matrix = 1 - cdist(vectors, vectors, metric="cosine")
    try:
        scores = degree_centrality_scores(
            similarity_matrix, threshold=threshold
        )
    except ValueError:
        scores = np.zeros(len(vocab))
    scores = dict(zip(vocab, scores))
    triple_score = {}
    for vocab_key, vocab_score in scores.items():
        triple_score[vocab2triple[vocab_key]] = vocab_score
    return triple_score


def score_triples_v3(
    triples: Dict[Tuple[str, str, str], set],
    embedding: Embedding,
    context: str,
    threshold: float = 0.1,
) -> Dict[Tuple[str, str, str], float]:
    vocab = {
        v.strip()
        for v in unibreak.split_words(context)
        if v.strip() and v not in stopwords
    }
    for s, p, o in triples.keys():
        vocab.update([s, o])
    vocab = list({v.lower() for v in vocab})
    ndim = embedding.shape[1]
    vectors = np.zeros((len(vocab), ndim))
    for i, term in enumerate(vocab):
        vectors[i] = embedding.get_embedding(term)
    similarity_matrix = 1 - cdist(vectors, vectors, metric="cosine")
    try:
        scores = degree_centrality_scores(
            similarity_matrix, threshold=threshold
        )
    except ValueError:
        scores = np.zeros(len(vocab))
    scores = dict(zip(vocab, scores))
    triple_score = {}
    for s, p, o in triples.keys():
        avg_score = (scores[s.lower()] + 2 * scores[o.lower()]) / 3
        triple_score[(s, p, o)] = avg_score
    return triple_score


def augment(text: str, triples: Dict[Tuple[str, str, str], set]):
    aug_text = text
    aug_triples = defaultdict(set)
    aug_triples.update(triples)
    # augmented is a mapping from the triplet representation
    #   to the character indices
    # it helps to avoid duplicating the same entity across
    #   different subjects of the sentence
    augmented = {}
    for triplet, _ in triples.items():
        # +1 is for the space
        triplet_repr = f" {triplet[1]} {triplet[2]}"
        if triplet_repr not in augmented:
            start = len(aug_text) + 1
            aug_text += triplet_repr
            end = len(aug_text)
            augmented[triplet_repr] = (start, end)
        else:
            start, end = augmented[triplet_repr]
        # j is the character index of the triplet/entity
        aug_triples[triplet].update(range(start, end + 1))
    return aug_text, aug_triples


def filter_triples(
    triples: Dict[Tuple[str, str, str], set],
    embedding: Embedding,
    top_k: int | None = None,
) -> Dict[Tuple[str, str, str], set]:
    if top_k is None:
        top_k = DEFAULT_TOP_K
    # select top 2 per subject
    scores = defaultdict(list)
    triplet_scores = score_triples(triples, embedding)
    for (s, p, o), score in triplet_scores.items():
        scores[s] += [(score, p, o)]
    filtered_triples = {}
    for s, scores in scores.items():
        for _, p, o in sorted(scores, reverse=True)[:top_k]:
            filtered_triples[(s, p, o)] = triples[(s, p, o)]
    return filtered_triples


def lexquery(lexicon: Lexicon, text: str) -> Dict[Tuple[str, str, str], set]:
    result = defaultdict(set)
    matches = list(lexicon.extract(text))
    visited = set()
    while matches:
        triples, start, end = matches.pop()
        for triple in triples:
            if triple in visited:
                continue
            visited.add(triple)
            _, p, o = triple
            if p in {"IsA", "HasContext"}:
                result[triple].update(range(start, end))
            elif p == "FormOf":
                matches.extend(
                    (forms, start, end) for forms, _, _ in lexicon.extract(o)
                )
    return result


def lexload(
    path: str | Path,
    case_sensitive: bool = False,
) -> Lexicon:
    path = Path(path)
    triplets = defaultdict(set)
    with open(path, "r") as f:
        nlines = sum(1 for _ in f)
    with open(path, "r") as f:
        reader = csv.reader(f)
        # no header
        for s, p, o in tqdm.tqdm(reader, total=nlines, desc="Loading triples"):
            k = preprocess(s)
            if k:
                triplets[k].add((s, p, o))
    lexicon = Lexicon(case_sensitive=case_sensitive)
    for k in triplets.keys():
        lexicon[k] = triplets[k]
    # build lexicon (no need to build from now on)
    lexicon.build()
    return lexicon
