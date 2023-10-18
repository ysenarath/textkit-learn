from typing import List, Union, Optional

import numpy as np

from tklearn.embedding.keyedvectors import KeyedVectors

__all__ = [
    "Tokenizer",
    "EmbeddingTokenizer",
]


class Tokenizer:
    def tokenize(self, texts: Union[str, List[str]]) -> List[List[str]]:
        raise NotImplementedError


class EmbeddingTokenizer(Tokenizer):
    def __init__(
        self,
        weights: KeyedVectors,
        padding_idx: Optional[int] = None,
        padding: Union[bool, str] = "max_length",
        truncation: bool = True,
        max_length: int = 200,
    ):
        # additional processing for additional_special_tokens
        if padding_idx is not None:
            weights = weights.insert(padding_idx, "[PAD]")
        self.weights = weights
        self.padding_idx = padding_idx
        self.padding = padding
        self.truncation = truncation
        self.max_length = max_length

    def pad_features(self, tokenized_texts):
        """Return features of tokenized_texts, where each review is padded with 0's
        or truncated to the input seq_length.
        """
        if self.padding == "max_length":
            padding_length = self.max_length
        elif self.padding == "longest" or self.padding is True:
            # get the longest length of the batch
            padding_length = np.max(list(map(len, tokenized_texts)))
        # getting the correct rows x cols shape
        features = np.zeros((len(tokenized_texts), padding_length), dtype=int)
        if self.padding_idx != 0:
            features[:] = self.padding_idx
        for i, row in enumerate(tokenized_texts):
            if not self.truncation and len(row) > padding_length:
                raise NotImplementedError
            features[i, -len(row) :] = np.array(row)[:padding_length]
        return features

    def tokenize(self, texts: List[str]):
        # split each review into a list of words
        sents = [text.split() for text in texts]
        tokenized_texts = []
        for tokens in sents:
            ints = []
            for token in tokens:
                try:
                    idx = self.weights.key_to_index[token]
                except:
                    if self.padding_idx is None:
                        continue
                    idx = self.padding_idx
                ints.append(idx)
            tokenized_texts.append(ints)
        if not self.padding:
            return tokenized_texts
        return self.pad_features(tokenized_texts)
