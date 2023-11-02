from __future__ import annotations
import functools
from typing import Any, List, Union, Optional, Dict
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from transformers import (
    AutoTokenizer,
    BatchEncoding,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)
from nltk.tokenize import WhitespaceTokenizer

from tklearn.core.model import Annotation, AnnotationList
from tklearn.embedding.keyedvectors import KeyedVectors

__all__ = [
    "Tokenizer",
    "EmbeddingTokenizer",
]

BatchAnnotationList = List[Union[AnnotationList, List[Annotation], List[Mapping]]]
TokenizerOutputType = Union[Mapping, Sequence, BatchEncoding]


class Tokenizer:
    def tokenize(self, *args: Any, **kwargs: Any) -> TokenizerOutputType:
        raise NotImplementedError

    def __call__(self, *args: Any, **kwargs: Any) -> TokenizerOutputType:
        return self.tokenize(*args, **kwargs)


class EmbeddingTokenizer(Tokenizer):
    """
    A custom tokenizer class for embedding-based tokenization.

    Parameters
    ----------
    weights : KeyedVectors
        The word embeddings used for tokenization.
    padding_idx : Optional[int], default None
        The index used for padding tokens. If not specified, no padding is applied.
    padding : Union[bool, str], default "max_length"
        Specifies the padding strategy. Can be "max_length," "longest," or True.
    truncation : bool, default True
        If True, truncate input sequences longer than the specified max_length.
    max_length : int, default 200
        The maximum length of tokenized sequences.

    Attributes
    ----------
    weights : KeyedVectors
        The word embeddings used for tokenization.
    padding_idx : Optional[int]
        The index used for padding tokens.
    padding : Union[bool, str]
        The padding strategy.
    truncation : bool
        Truncation flag.
    max_length : int
        The maximum length of tokenized sequences.
    _nltk_tokenizer : WhitespaceTokenizer
        An NLTK tokenizer used for splitting text into words.

    Methods
    -------
    pad(input_ids: List[List[int]]) -> np.ndarray
        Return features of tokenized texts, where each review is padded with 0's
        or truncated to the input seq_length.

    tokenize(texts: Union[List[str], str]) -> BatchEncoding
        Tokenize a list of texts and return a BatchEncoding object.

    Example
    -------
    >>> embeddings = KeyedVectors.load_word2vec_format('path_to_embeddings.bin', binary=True)
    >>> tokenizer = EmbeddingTokenizer(weights=embeddings, padding=True, max_length=256)
    >>> texts = ["This is a sample sentence.", "Another example for tokenization."]
    >>> encodings = tokenizer.tokenize(texts)
    """

    def __init__(
        self,
        weights: KeyedVectors,
        padding_idx: Optional[int] = None,
        padding: Union[bool, str] = "max_length",
        truncation: bool = True,
        max_length: int = 200,
    ):
        """
        Initialize the EmbeddingTokenizer with the specified parameters.

        Parameters
        ----------
        weights : KeyedVectors
            The word embeddings used for tokenization.
        padding_idx : Optional[int], default None
            The index used for padding tokens. If not specified, no padding is applied.
        padding : Union[bool, str], default "max_length"
            Specifies the padding strategy. Can be "max_length," "longest," or True.
        truncation : bool, default True
            If True, truncate input sequences longer than the specified max_length.
        max_length : int, default 200
            The maximum length of tokenized sequences.
        """
        # additional processing for additional_special_tokens
        if padding_idx is not None:
            weights = weights.insert(padding_idx, "[PAD]")
        self.weights = weights
        self.padding_idx = padding_idx
        self.padding = padding
        self.truncation = truncation
        self.max_length = max_length
        self._nltk_tokenizer = WhitespaceTokenizer()

    def pad(
        self,
        input_ids: List[List[int]],
    ) -> np.ndarray:
        """
        Return features of tokenized_texts, where each review is padded with provided
        padding_idx or truncated to the input seq_length.

        Parameters
        ----------
        input_ids : List[List[int]]
            A list of input sequences, each represented as a list of integers.

        Returns
        -------
        np.ndarray
            An array of padded or truncated sequences.
        """
        """Return features of tokenized_texts, where each review is padded with 0's
        or truncated to the input seq_length.
        """
        if self.padding == "max_length":
            padding_length = self.max_length
        elif self.padding == "longest" or self.padding is True:
            # get the longest length of the batch
            padding_length = np.max(list(map(len, input_ids)))
        # getting the correct rows x cols shape
        features = np.zeros((len(input_ids), padding_length), dtype=int)
        if self.padding_idx != 0:
            features[:] = self.padding_idx
        for i, row in enumerate(input_ids):
            if not self.truncation and len(row) > padding_length:
                raise NotImplementedError
            features[i, -len(row) :] = np.array(row)[:padding_length]
        return features

    def tokenize(
        self,
        texts: Union[List[str], str],
    ) -> BatchEncoding:
        """
        Tokenize a list of texts and return a BatchEncoding object.

        Parameters
        ----------
        texts : Union[List[str], str]
            A single text or a list of texts to tokenize.

        Returns
        -------
        BatchEncoding
            A BatchEncoding object containing the tokenized input.

        Notes
        -----
        This method splits each text into a list of words and assigns word embeddings to tokens.
        """
        # split each review into a list of words
        if isinstance(texts, str):
            texts = [texts]
        input_ids = []
        for text in texts:
            token_spans = self._nltk_tokenizer.span_tokenize(text)
            ints = []
            char_to_token = ([], [])
            for token_span in token_spans:
                token = text[slice(*token_span)]
                try:
                    idx = self.weights.key_to_index[token]
                except:
                    if self.padding_idx is None:
                        continue
                    idx = self.padding_idx
                token_idx = len(ints)
                ints.append(idx)
                char_to_token[0].extend(token_span)
                char_to_token[1].append(token_idx)
            input_ids.append(ints)
        if not self.padding:
            return BatchEncoding({"input_ids": input_ids})
        return BatchEncoding({"input_ids": self.pad(input_ids)})


class HuggingFaceTokenizer(Tokenizer):
    """
    A custom tokenizer class that wraps the Hugging Face tokenizer.

    Parameters
    ----------
    tokenizer : Union[PreTrainedTokenizer, PreTrainedTokenizerFast]
        The Hugging Face tokenizer to wrap.

    Methods
    -------
    tokenize(*args, **kwargs) -> BatchEncoding
        Tokenize input text using the wrapped Hugging Face tokenizer.

    get_aligned_labels(encoding: BatchEncoding, annotations: AnnotationListType) -> Dict[str, List[int]]
        Align labels to tokens based on annotations and return a dictionary of aligned label lists.

    from_pretrained(*args, **kwargs) -> 'HuggingFaceTokenizer'
        Create an instance of the HuggingFaceTokenizer class from a pretrained Hugging Face tokenizer.

    Attributes
    ----------
    _hf_tokenizer : functools.partial
        The wrapped Hugging Face tokenizer.

    Example
    -------
    >>> hft = HuggingFaceTokenizer.from_pretrained("bert-base-uncased")
    >>> encoding = hft.tokenize("Hello, world!")
    >>> labels = hft.get_aligned_labels(encoding, annotations)
    """

    def __init__(
        self,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        *args,
        **kwargs,
    ):
        """
        Initialize the HuggingFaceTokenizer with a Hugging Face tokenizer.

        Parameters
        ----------
        tokenizer : Union[PreTrainedTokenizer, PreTrainedTokenizerFast]
            The Hugging Face tokenizer to wrap.
        """
        self._hf_tokenizer: functools.partial = functools.partial(
            tokenizer, *args, **kwargs
        )

    def tokenize(self, *args, **kwargs) -> BatchEncoding:
        """
        Tokenize input text using the wrapped Hugging Face tokenizer.

        Returns
        -------
        BatchEncoding
            The tokenized input.

        Parameters
        ----------
        *args
            Variable-length positional arguments to pass to the underlying tokenizer.
        **kwargs
            Variable-length keyword arguments to pass to the underlying tokenizer.
        """
        return self._hf_tokenizer(*args, **kwargs)

    def get_aligned_labels(
        self,
        batch_encoding: Union[
            BatchEncoding,
            List[BatchEncoding],
            pd.DataFrame,
            pd.Series,
        ],
        batch_annotations: BatchAnnotationList,
    ) -> Dict[str, List[int]]:
        """
        Align labels to tokens based on annotations and return a dictionary of aligned label lists.

        Returns
        -------
        Dict[str, List[int]]
            A dictionary where keys are labels and values are lists of aligned labels for each token.

        Parameters
        ----------
        encoding : BatchEncoding
            The tokenized input encoding.
        annotations : AnnotationListType
            A list of annotations to align with tokens.
        """
        # Initialize a dictionary to store lists of aligned labels for each label value
        batch_aligned_labels = []
        if isinstance(batch_encoding, BatchEncoding):
            batch_size = len(batch_encoding._encodings)
        elif isinstance(batch_encoding, (pd.DataFrame, pd.Series)):
            batch_size = len(batch_encoding)
        elif isinstance(batch_encoding, Sequence) and not isinstance(
            batch_encoding, str
        ):
            batch_size = len(batch_encoding)
        else:
            raise TypeError(
                f"unsupported type for 'batch_encoding': {type(batch_encoding)}"
            )
        for batch_index in range(batch_size):
            annotations = batch_annotations[batch_index]
            if isinstance(batch_encoding, BatchEncoding):
                sample_index = batch_index
                sample_encoding = batch_encoding
            elif isinstance(batch_encoding, pd.DataFrame):
                sample_index = 0
                sample_encoding = batch_encoding.iloc[batch_index]
            else:  # batch encoding is a list of batch encoding type
                sample_index = 0
                sample_encoding = batch_encoding[batch_index]
            sequence_len = len(sample_encoding.tokens(batch_index=sample_index))
            aligned_labels = {}
            # Iterate through annotations and align labels to tokens
            for anno_idx, anno in enumerate(annotations):
                if isinstance(anno, Annotation):
                    anno = {
                        "start": anno.span.start,
                        "end": anno.span.end,
                        "label": anno.label,
                    }
                elif not isinstance(anno, Mapping):
                    raise TypeError(
                        f"annotation must be a dict or Annotation type, "
                        f"found {anno.__class__.__name__}"
                    )
                token_position = 1
                span = anno.get("span", anno)
                label = anno.get("label", None)
                for char_ix in range(span["start"], span["end"]):
                    token_ix = sample_encoding.char_to_token(sample_index, char_ix)
                    if token_ix is None:
                        continue
                    # Initialize a list for the label if it doesn't exist
                    if label not in aligned_labels:
                        aligned_labels[label] = [(0, -1)] * sequence_len
                    if aligned_labels[label][token_ix][0] == 0:
                        aligned_labels[label][token_ix] = token_position, anno_idx
                    token_position = 2
            batch_aligned_labels.append(aligned_labels)
        return batch_aligned_labels

    @classmethod
    def from_pretrained(
        cls, pretrained_model_or_path: str, *args, **kwargs
    ) -> HuggingFaceTokenizer:
        """
        Create an instance of the HuggingFaceTokenizer class from a pretrained Hugging Face tokenizer.

        Returns
        -------
        HuggingFaceTokenizer
            An instance of the HuggingFaceTokenizer class.

        Parameters
        ----------
        *args
            Variable-length positional arguments to pass to the `AutoTokenizer.from_pretrained` method.
        **kwargs
            Variable-length keyword arguments to pass to the `AutoTokenizer.from_pretrained` method.
        """
        hft = AutoTokenizer.from_pretrained(pretrained_model_or_path)
        return cls(hft, *args, **kwargs)
