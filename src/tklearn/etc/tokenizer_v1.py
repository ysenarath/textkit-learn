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
from transformers import AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from typing_extensions import Self

from tklearn.embeddings import AutoEmbedding
from tklearn.etc.helpers import SubwordDetector
from tklearn.utils.lexrank import degree_centrality_scores
from tklearn.utils.trie import BytesTrie

__all__ = ["KnowledgeBasedTokenizer"]

nltk.download("wordnet")
nltk.download("punkt_tab")
nltk.download("stopwords")

DEFAULT_TOP_K = 3


class KnowledgeBasedTokenizer:
    tokenizer: PreTrainedTokenizer
    triplets: BytesTrie
    punctrans: str
    stopwords: set
    subword_detector: SubwordDetector
    embedding: AutoEmbedding

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
        self.triplets = None
        self.punctrans = str.maketrans("", "", string.punctuation)
        self.stopwords = set(sw.words("english"))
        # has the tokenizer been created?
        # AutoTokenizer.from_pretrained("bert-base-uncased")
        self.subword_detector = SubwordDetector(self.tokenizer)
        self.embedding = AutoEmbedding.from_config({"identifier": "fasttext"})
        return self

    def load_triples(self, path: str | Path) -> None:
        path = Path(path)
        cache_path = path.with_suffix(".marisa")
        if cache_path.exists():
            self.triplets = BytesTrie().mmap(str(cache_path))
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
                for _, p, _ in pickle.loads(triple):
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
        for i, index in enumerate(ids):
            index = int(index)
            if skip_special_tokens and index in self.tokenizer.all_special_ids:
                continue
            if index in getattr(self.tokenizer, "_added_tokens_decoder", {}):
                content = self.tokenizer._added_tokens_decoder[index].content
            else:
                content = self.tokenizer._convert_id_to_token(index)
            tokens.append((i, content))
        return tokens

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        return self.tokenizer.convert_tokens_to_string(tokens)

    def preprocess(self, text: str) -> str:
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

    def query(self, text: str) -> Dict[Tuple[str, str, str], set]:
        encoding = self.tokenizer(text, return_offsets_mapping=True)
        input_ids = encoding["input_ids"]
        offsets = encoding["offset_mapping"]
        local_triples = defaultdict(set)
        tokens: List[Tuple[int, str]] = self.convert_ids_to_tokens(
            input_ids,
            skip_special_tokens=True,
        )
        i = 0
        while i < len(tokens):
            start_token = tokens[i][1]
            if self.subword_detector.is_prefix(start_token, i == 0):
                # special token or subword
                i += 1
                continue
            token_triples = defaultdict(set)
            for j in range(i + 1, len(tokens) + 1):
                k = self.convert_tokens_to_string([i[1] for i in tokens[i:j]])
                k = self.preprocess(k)
                if not k:
                    # not worth exploring - go to the next token
                    break
                if not self.triplets.keys(k):
                    # did not find a prefix
                    # no need to explore further
                    # first add the current triples to the local triples
                    for key, value in token_triples.items():
                        local_triples[key].update(value)
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
                token_triples.clear()
                for match in matches:
                    # edges is the set of tuples
                    edges = pickle.loads(match)
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
                        if edge[1] in {"FormOf"}:
                            if depth > 0:
                                # only look at forms of one hop away
                                continue
                            # extend the edges with the triplets of the entity
                            extended_matches = self.triplets.get(edge[2])
                            if extended_matches is None:
                                continue
                            for extended_match in extended_matches:
                                extended_edges = pickle.loads(extended_match)
                                edges.update({
                                    (depth + 1, e) for e in extended_edges
                                })
                            continue
                        # go to the next edge if the edge is not IsA or HasContext
                        if edge[1] not in {"IsA", "HasContext"}:
                            continue
                        token_triples[edge].update(
                            range(
                                offsets[tokens[i][0]][0],
                                offsets[tokens[j - 1][0]][1],
                            )
                        )
            i += 1
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
        for text in texts:
            triples = self.query(text)
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
        for i in range(len(texts)):
            aug_start, aug_triples = augmented[1][i]
            offsets = encodings["offset_mapping"][0]
            num_tokens = len(offsets)
            token_triples = self.get_entites_per_token(offsets, aug_triples)
            # visibility matrix
            M = np.zeros((num_tokens, num_tokens), dtype=int)
            for i in range(num_tokens):
                for j in range(num_tokens):
                    if (
                        offsets[j][0].item() <= aug_start
                        and offsets[i][1].item() <= aug_start
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
        vectors = {}
        vocab = set()
        for s, v, o in triples.keys():
            vocab.update([s, o])
        vocab = list(vocab)
        for term in vocab:
            vectors[term] = self.embedding.get_word_vector(term)
        similarity_matrix = np.zeros((len(vocab), len(vocab)))
        for i, term in enumerate(vocab):
            for j, other_term in enumerate(vocab):
                if term == other_term:
                    continue
                a, b = vectors[term], vectors[other_term]
                denom = np.linalg.norm(a) * np.linalg.norm(b)
                if denom == 0:
                    s = 0
                else:
                    s = np.dot(a, b) / (denom)
                similarity_matrix[i, j] = s
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
