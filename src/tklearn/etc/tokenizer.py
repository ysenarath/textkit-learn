from __future__ import annotations

import csv
import pickle
import string
from collections import defaultdict
from os import PathLike
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import tqdm
import nltk
from nltk.corpus import stopwords as sw
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from transformers import AutoTokenizer, PreTrainedTokenizer
from typing_extensions import Self

from tklearn.utils.trie import BytesTrie

__all__ = ["KnowledgeBasedTokenizer"]

nltk.download("wordnet")


class KnowledgeBasedTokenizer:
    def __init__(self, triplets: BytesTrie = None):
        self.triplets = triplets
        self.punctrans = str.maketrans("", "", string.punctuation)
        self.stopwords = set(sw.words("english"))
        self.tokenizer: PreTrainedTokenizer = getattr(
            self, "tokenizer", self._create_tokenizer()
        )

    @classmethod
    def _create_tokenizer(cls) -> PreTrainedTokenizer:
        return AutoTokenizer.from_pretrained("bert-base-uncased")

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str | PathLike, **kwargs
    ) -> Self:
        self = cls.__new__(cls)
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path
        )
        self.__init__(**kwargs)
        return self

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
        # remove all stopwords
        tokens = word_tokenize(s)
        # lemmatize
        tokens = [
            (i if i in self.stopwords else (wn.morphy(i) or i)) for i in tokens
        ]
        return " ".join(tokens).strip()

    def query(self, text: str):
        input_obj = self.tokenizer(text, return_offsets_mapping=True)
        input_ids = input_obj["input_ids"]
        offsets = input_obj["offset_mapping"]
        entities = defaultdict(set)
        tokens = self.convert_ids_to_tokens(input_ids)
        for i in range(len(tokens)):
            current_entities = defaultdict(set)
            for j in range(i + 1, len(tokens) + 1):
                k = self.convert_tokens_to_string([i[1] for i in tokens[i:j]])
                # print(k, end=", ")
                k = self.preprocess(k)
                # print(k, end=", ")
                # print()
                if not k:
                    # not worth exploring
                    break
                if not self.triplets.keys(k):
                    # did not find a prefix
                    # no need to explore further
                    # first add the current entities to the global entities
                    for key, value in current_entities.items():
                        entities[key].update(value)
                    break
                matches = self.triplets.get(k)
                if matches is None:
                    # may be we found a prefix
                    continue
                for match in matches:
                    for edge in pickle.loads(match):
                        current_entities[edge].update(
                            range(
                                offsets[tokens[i][0]][0],
                                offsets[tokens[j - 1][0]][1] + 1,
                            )
                        )
        return entities

    def augment(self, text: str, entities: dict):
        aug_text = text
        aug_entities = defaultdict(set)
        aug_entities.update(entities)
        for triplet, _ in entities.items():
            # +1 is for the space
            start = len(aug_text) + 1
            aug_text += f" {triplet[1]} {triplet[2]}"
            end = len(aug_text)
            # j is the character index of the triplet/entity
            aug_entities[triplet].update(range(start, end + 1))
        return aug_text, aug_entities

    def get_entites_per_token(
        self,
        offsets: List[Tuple[int, int]],
        entities: Dict[Tuple[str, str, str], set],
    ):
        ntokens = len(offsets)
        token_entities = defaultdict(set)
        for i in range(ntokens):
            # both mention and token are sets of character indices
            token_char_idxs = set(range(*offsets[i]))
            for entity, mention_char_idxs in entities.items():
                # if any of the mention chars are in the token chars
                if token_char_idxs.intersection(mention_char_idxs):
                    token_entities[i].add(entity)
        return token_entities

    def tokenize(
        self,
        text: str | List[str],
        return_offsets_mapping: bool = False,
        return_tokens: bool = False,
    ) -> Dict[str, torch.Tensor | List[List[str]]]:
        texts = [text] if isinstance(text, str) else text
        del text
        augmented = ([], [])
        for text in texts:
            entities = self.query(text)
            aug_start = len(text)
            aug_text, aug_entities = self.augment(text, entities)
            augmented[0].append(aug_text)
            augmented[1].append((aug_start, aug_entities))
        encodings = self.tokenizer(
            augmented[0],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            return_offsets_mapping=True,
        )
        visibility_mask = []
        for i in range(len(texts)):
            aug_start, aug_entities = augmented[1][i]
            offsets = encodings["offset_mapping"][0]
            num_tokens = len(offsets)
            token_entities = self.get_entites_per_token(offsets, aug_entities)
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
                    shared_entities = token_entities[i].intersection(
                        token_entities[j]
                    )
                    if shared_entities:
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
    ):
        return self.tokenize(
            text,
            return_offsets_mapping=return_offsets_mapping,
            return_tokens=return_tokens,
        )

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
