from __future__ import annotations

import csv
import pickle
import string
from collections import defaultdict
from os import PathLike
from pathlib import Path
from typing import Dict, List, Tuple

import nltk
import numpy as np
import torch
import tqdm
from nltk.corpus import stopwords as sw
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from scipy.spatial.distance import cdist
from transformers import AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from typing_extensions import Self

from tklearn.embeddings import AutoEmbedding, Embedding
from tklearn.etc.helpers import SubwordDetector
from tklearn.utils.lexrank import degree_centrality_scores
from tklearn.utils.trie import BytesTrie

__all__ = ["KnowledgeBasedTokenizer"]

nltk.download("wordnet", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

DEFAULT_TOP_K = 3

# TODO:
#  - set notation may be replaced with intervaltree


class KnowledgeBasedTokenizer:
    tokenizer: PreTrainedTokenizer
    punctrans: str
    stopwords: set
    subword_detector: SubwordDetector
    embedding: Embedding

    def __init__(self):
        raise NotImplementedError

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str | PathLike, **kwargs
    ) -> Self:
        self = cls.__new__(cls)
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path
        )
        self.punctrans = str.maketrans("", "", string.punctuation)
        self.stopwords = set(sw.words("english"))
        # has the tokenizer been created?
        # AutoTokenizer.from_pretrained("bert-base-uncased")
        self.subword_detector = SubwordDetector(self.tokenizer)
        self.embedding = AutoEmbedding.from_config({"identifier": "fasttext"})
        return self

    @property
    def triplets(self) -> BytesTrie:
        triples = getattr(self, "_triplets", None)
        if triples is None:
            raise AttributeError("triplets have not been set")
        return triples

    @triplets.setter
    def triplets(self, value: BytesTrie):
        self._triplets = value
        try:
            self.load_triplet(b"")
        except KeyError:
            pass

    def load_triplet(self, key: bytes) -> List[Tuple[str, str, str]]:
        if not hasattr(self, "_pickle_cache"):
            cache = {}
            for k in self.triplets.keys():
                for triple in self.triplets.get(k):
                    cache[triple] = pickle.loads(triple)
            self._pickle_cache = cache
        return self._pickle_cache[key]

    def load_triples(self, path: str | Path) -> None:
        path = Path(path)
        cache_path = path.with_suffix(".marisa")
        if cache_path.exists():
            # self.triplets = BytesTrie().mmap(str(cache_path))
            triplets = BytesTrie()
            triplets.load(str(cache_path))
            self.triplets = triplets
            return
        triplets = defaultdict(set)
        with open(path, "r") as f:
            nlines = sum(1 for _ in f)
        with open(path, "r") as f:
            reader = csv.reader(f)
            # no header
            for s, p, o in tqdm.tqdm(
                reader, total=nlines, desc="Loading triples"
            ):
                k = self.preprocess(s)
                triplets[k].add((s, p, o))
        triplets = ((k, pickle.dumps(v)) for k, v in triplets.items())
        triplets = BytesTrie(triplets)
        # cache it near the path for future use
        triplets.save(str(cache_path))
        self.triplets = triplets

    def prepare_model(self, model: PreTrainedModel) -> PreTrainedModel:
        new_tokens = set()
        for k in self.triplets.keys():
            for triple in self.triplets.get(k):
                for _, p, _ in self.load_triplet(triple):
                    new_tokens.add(p)
        vocab = set(self.tokenizer.get_vocab().keys())
        new_tokens = new_tokens - vocab
        self.tokenizer.add_tokens(list(new_tokens))
        model.resize_token_embeddings(len(self.tokenizer))
        return model

    def convert_ids_to_tokens(
        self, ids: List[int], skip_special_tokens: bool = True
    ) -> List[Tuple[int, str]]:
        tokens = []
        for i, id_ in enumerate(ids):
            id_ = int(id_)
            if skip_special_tokens and id_ in self.tokenizer.all_special_ids:
                continue
            if id_ in getattr(self.tokenizer, "_added_tokens_decoder", {}):
                content = self.tokenizer._added_tokens_decoder[id_].content
            else:
                content = self.tokenizer._convert_id_to_token(id_)
            tokens.append((i, content))
        return tokens

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        return self.tokenizer.convert_tokens_to_string(tokens)

    def preprocess(self, text: str) -> str:
        # this could have drastic effects on the input sentence
        #   therefore should not be used on the full text
        s = text.lower()
        # remove all consecative spaces
        s = " ".join(s.split())
        # remove all punctuations
        s = s.translate(self.punctrans)
        # remove all digits (by checking if each character is a digit)
        s = "".join([i for i in s if not i.isdigit()])
        # tokenize
        tokens = word_tokenize(s)
        # lemmatize
        tokens = [
            (i if i in self.stopwords else (wn.morphy(i) or i)) for i in tokens
        ]
        return " ".join(tokens).strip()

    def query(
        self, encodings: dict, index: int
    ) -> Dict[Tuple[str, str, str], set]:
        input_ids = encodings["input_ids"][index]
        offsets = encodings["offset_mapping"][index]
        local_triples = defaultdict(set)
        tokens = self.convert_ids_to_tokens(input_ids)
        i = 0
        while i < len(tokens):  # for each instance
            start_token = tokens[i][1]
            if self.subword_detector.is_prefix(start_token, i == 0):
                # special token or subword
                i += 1
                continue
            phrase_triples = defaultdict(set)
            add_to_i = 1
            for j in range(i + 1, len(tokens) + 1):
                k = self.convert_tokens_to_string([i[1] for i in tokens[i:j]])
                k = self.preprocess(k)
                if not k:
                    # not worth exploring - go to the next token
                    break
                if not self.triplets.keys(k):
                    # did not find a prefix
                    # no need to explore further
                    add_to_i = j - i - 1
                    break
                if k in self.stopwords:
                    # do not match stopwords
                    continue
                matches = self.triplets.get(k)
                if matches is None:
                    # may be we found a prefix
                    continue
                # there are matches longer than the current token sequence
                #   we need to update the token triples
                phrase_triples.clear()
                for match in matches:
                    # edges is the set of tuples
                    edges = self.load_triplet(match)
                    # visited will help to avoid cycles
                    visited = set()
                    edges = {(0, edge) for edge in edges}
                    while len(edges) > 0:
                        depth, edge = edges.pop()
                        # same edge may only be visited once
                        if edge in visited:
                            continue
                        # mark the edge as visited
                        visited.add(edge)
                        # if the edge is a FormOf edge
                        #   only depth of <= 1 are allowed
                        #   we check < 1 because we want to extend depth 0 but not 1
                        if depth < 1 and edge[1] in {"FormOf"}:  # fix - FormOf
                            # extend the edges with the triplets of the entity
                            #   edge[2] is the object of the edge
                            forms = self.triplets.get(edge[2])
                            if forms is None:
                                continue
                            for form in forms:
                                form_triplets = self.load_triplet(form)
                                edges.update({
                                    (depth + 1, e) for e in form_triplets
                                })
                            continue
                        # go to the next edge if the edge is not IsA or HasContext
                        #   edge[1] is the predicate of the edge
                        if edge[1] not in {"IsA", "HasContext"}:
                            continue
                        phrase_triples[edge].update(
                            range(
                                offsets[tokens[i][0]][0],
                                offsets[tokens[j - 1][0]][1],
                            )
                        )
            # first add the current triples to the local triples
            for key, value in phrase_triples.items():
                local_triples[key].update(value)
            i += add_to_i
        return local_triples

    def augment(self, text: str, triples: Dict[Tuple[str, str, str], set]):
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

    def get_entites_per_token(
        self,
        offsets: List[Tuple[int, int]],
        triples: Dict[Tuple[str, str, str], set],
    ):
        ntokens = len(offsets)
        token_triples = defaultdict(set)
        for i in range(ntokens):
            # both mention and token are sets of character indices
            token_char_idxs = set(range(*offsets[i]))
            for triple, mention_char_idxs in triples.items():
                # if any of the mention chars are in the token chars
                if token_char_idxs.intersection(mention_char_idxs):
                    token_triples[i].add(triple)
        return token_triples

    def tokenize(
        self,
        text: str | List[str],
        return_offsets_mapping: bool = False,
        return_tokens: bool = False,
        top_k: int = DEFAULT_TOP_K,
    ) -> Dict[str, torch.Tensor | List[List[str]]]:
        texts = [text] if isinstance(text, str) else text
        del text
        augmented = ([], [])
        encodings = self.tokenizer(texts, return_offsets_mapping=True)
        for i, text in enumerate(texts):
            triples = self.query(encodings, i)
            triples = self.filter(triples, top_k=top_k)
            aug_start = len(text)
            aug_text, aug_triples = self.augment(text, triples)
            augmented[0].append(aug_text)
            augmented[1].append((aug_start, aug_triples))
        encodings = self.tokenizer(
            augmented[0],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            return_offsets_mapping=True,
        )
        visibility_mask = []
        offset_mapping = encodings["offset_mapping"].cpu().numpy()
        for index in range(len(texts)):
            aug_start, aug_triples = augmented[1][index]
            offsets = offset_mapping[index]
            num_tokens = len(offsets)
            token_triples = self.get_entites_per_token(offsets, aug_triples)
            # visibility matrix
            M = np.zeros((num_tokens, num_tokens), dtype=int)
            for i in range(num_tokens):
                for j in range(num_tokens):
                    if (
                        offsets[j][0] <= aug_start
                        and offsets[i][1] <= aug_start
                    ):
                        M[i][j] = 1
                        continue
                    shared_triples = token_triples[i].intersection(
                        token_triples[j]
                    )
                    if shared_triples:
                        M[i][j] = 1
            visibility_mask.append(torch.from_numpy(M))
        encodings["visibility_mask"] = torch.stack(visibility_mask)
        if not return_offsets_mapping:
            encodings.pop("offset_mapping")
        if return_tokens:
            encodings["tokens"] = [
                self.tokenizer.convert_ids_to_tokens(input_ids)
                for input_ids in encodings["input_ids"]
            ]
        return encodings

    def __call__(
        self,
        text: str | List[str],
        return_offsets_mapping: bool = False,
        return_tokens: bool = False,
        top_k: int = DEFAULT_TOP_K,
    ):
        return self.tokenize(
            text,
            return_offsets_mapping=return_offsets_mapping,
            return_tokens=return_tokens,
            top_k=top_k,
        )

    def filter(
        self,
        triples: Dict[Tuple[str, str, str], set],
        top_k: int = DEFAULT_TOP_K,
    ) -> Dict[Tuple[str, str, str], set]:
        # select top 2 per subject
        scores = defaultdict(list)
        for (s, p, o), score in self.get_scores(triples).items():
            scores[s] += [(score, p, o)]
        filtered_triples = {}
        for s, scores in scores.items():
            for _, p, o in sorted(scores, reverse=True)[:top_k]:
                filtered_triples[(s, p, o)] = triples[(s, p, o)]
        return filtered_triples

    def get_scores(
        self, triples: Dict[Tuple[str, str, str], set]
    ) -> List[Tuple[str, int]]:
        vocab = set()
        for s, v, o in triples.keys():
            vocab.update([s, o])
        vocab = list(vocab)
        ndim = self.embedding.shape[1]
        vectors = np.zeros((len(vocab), ndim))
        for i, term in enumerate(vocab):
            vectors[i] = self.embedding.get_word_vector(term)
        similarity_matrix = 1 - cdist(vectors, vectors, metric="cosine")
        try:
            scores = degree_centrality_scores(similarity_matrix, threshold=0.1)
        except ValueError:
            scores = np.zeros(len(vocab))
        scores = dict(zip(vocab, scores))
        triple_score = {}
        for s, v, o in triples.keys():
            avg_score = (scores[s] + scores[o]) / 2
            triple_score[(s, v, o)] = avg_score
        return triple_score
