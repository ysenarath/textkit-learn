from __future__ import annotations

from collections import defaultdict
from os import PathLike
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from typing_extensions import Self

from tklearn.embeddings import AutoEmbedding, Embedding
from tklearn.kb.lexicon import Lexicon
from tklearn.nn.models.backbone.knowledge import helpers

__all__ = [
    "KnowledgeBasedTokenizer",
]


class KnowledgeBasedTokenizer:
    tokenizer: PreTrainedTokenizer
    embedding: Embedding
    lexicon: Lexicon[Set[Tuple[str, str, str]]]

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
        # has the tokenizer been created?
        self.embedding = AutoEmbedding.from_config({"name": "fasttext"})
        return self

    def load_triples(self, path: str | Path) -> None:
        self.lexicon = helpers.lexload(path)

    def prepare_model(self, model: PreTrainedModel) -> PreTrainedModel:
        new_tokens = set()
        for triples in self.lexicon.values():
            for _, p, _ in triples:
                new_tokens.add(p)
        vocab = set(self.tokenizer.get_vocab().keys())
        new_tokens = new_tokens - vocab
        self.tokenizer.add_tokens(list(new_tokens))
        model.resize_token_embeddings(len(self.tokenizer))
        return model

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

    def analyze(self, text: str, top_k: int | None = None):
        triples = self.lexicon.extract(text)
        if top_k:
            triples = helpers.filter_triples(
                triples,
                top_k=top_k,
                embedding=self.embedding,
            )
        return triples

    def encode(
        self,
        text: str | List[str],
        return_offsets_mapping: bool = False,
        return_tokens: bool = False,
        top_k: int | None = None,
    ) -> Dict[str, torch.Tensor | List[List[str]]]:
        texts = [text] if isinstance(text, str) else text
        augmented = ([], [])
        for i, text in enumerate(texts):
            triples = helpers.lexquery(self.lexicon, text)
            triples = helpers.filter_triples(
                triples, embedding=self.embedding, top_k=top_k
            )
            aug_start = len(text)
            aug_text, aug_triples = helpers.augment(text, triples)
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
        top_k: int | None = None,
    ):
        return self.encode(
            text,
            return_offsets_mapping=return_offsets_mapping,
            return_tokens=return_tokens,
            top_k=top_k,
        )
