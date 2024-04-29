"""Annotator module for keyword extraction and alignment.

This module provides a class for annotating text with keywords and labels.

Examples
--------
>>> from tklearn.preprocessing.annotator import KeywordAnnotator
>>> from tklearn.base.resource import ResourceIO
>>> from tklearn.kb.emolex.io import EmoLexIO
>>> io = EmoLexIO()
>>> emolex = KeywordAnnotator(io, "word", "emotion")
>>> prompt = ["this is a happy family", "this is a sad family"]
>>> bencoding = tokenizer.tokenize(
...     prompt,
...     truncation=True,
...     padding="max_length",
... )
>>> emolex.annotate(prompt, bencoding, return_tensors="pt")
"""

from typing import Dict, List, Optional, Union

import numpy as np
import torch
from flashtext import KeywordProcessor
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from tokenizers import Encoding
from tqdm import auto as tqdm
from transformers.tokenization_utils_base import BatchEncoding

from tklearn.base.resource import ResourceIO

__all__ = [
    "KeywordAnnotator",
]


def align(
    tokenized: Encoding,
    annotations: List[Dict],
    labels: Optional[List[str]] = None,
    return_tensors: Optional[str] = None,
) -> Union[List[dict], List[List[dict]]]:
    tokens = tokenized.tokens
    aligned_labels = {}
    aligned_scores = {}
    for anno in annotations:
        anno_label = anno["label"]
        annotation_token_ix_set = set()
        for char_ix in range(anno["start"], anno["end"]):
            token_ix = tokenized.char_to_token(char_ix)
            if token_ix is not None:
                annotation_token_ix_set.add(token_ix)
        for num, token_ix in enumerate(sorted(annotation_token_ix_set)):
            if num == 0:
                prefix = "B"
            else:
                prefix = "I"  # We're inside of a multi token annotation
            if anno_label not in aligned_labels:
                aligned_labels[anno_label] = ["O" for _ in range(len(tokens))]
                aligned_scores[anno_label] = np.zeros(len(tokens))
            aligned_labels[anno_label][token_ix] = prefix
            aligned_scores[anno_label][token_ix] = anno["score"]
    if labels is None:
        tok_anns = [[] for _ in range(len(tokens))]
        for anno_label in aligned_labels.keys():
            for i, (prefix, score) in enumerate(
                zip(aligned_labels[anno_label], aligned_scores[anno_label])
            ):
                tok_anns[i].append({
                    "label": f"{prefix}-{anno_label}",
                    "score": score,
                })
        return tok_anns
    tok_anns = [[0.0 for _ in range(len(labels))] for _ in range(len(tokens))]
    for anno_label in aligned_labels.keys():
        for i, (prefix, score) in enumerate(
            zip(aligned_labels[anno_label], aligned_scores[anno_label])
        ):
            if prefix == "O":
                continue
            tok_anns[i][labels.index(anno_label)] = score
    if return_tensors == "pt":
        return torch.tensor(tok_anns)
    elif return_tensors == "np":
        return np.array(tok_anns)
    return tok_anns


def batch_align(
    tokenized: BatchEncoding,
    annotations: List[Dict],
    labels: Optional[List[str]] = None,
    return_tensors: Optional[str] = None,
) -> Union[List[List[dict]], List[List[List[dict]]]]:
    if labels is None and return_tensors is not None:
        msg = "return_tensors must be None if labels is None"
        raise ValueError(msg)
    aligned = [
        align(tokenized[i], annotation, labels)
        for i, annotation in enumerate(annotations)
    ]
    if return_tensors == "pt":
        return torch.tensor(aligned)
    elif return_tensors == "np":
        return np.array(aligned)
    return aligned


class KeywordAnnotator:
    def __init__(
        self,
        io: ResourceIO,
        keyword_field: str,
        label_field: Union[str, List[str]],
        labels: Union[List, bool, None] = None,
    ):
        io.download(exist_ok=True, unzip=True)
        namespace = ".".join([io.__module__, io.__class__.__name__])
        self.resource = list(io.load())
        self.label_fields = (
            [label_field] if isinstance(label_field, str) else label_field
        )
        collected_labels = set() if labels is True else None
        for doc in tqdm.tqdm(self.resource):
            if collected_labels is None:
                continue
            for label_field in self.label_fields:
                for label in doc[label_field].keys():
                    collected_labels.add(f"{namespace}/{label_field}/{label}")
        self.labels = (
            labels if collected_labels is None else sorted(list(collected_labels))
        )
        self.namespace = namespace
        self.keyword_field = keyword_field

    def get_keyword_processor(self, language: str = "english") -> KeywordProcessor:
        if not hasattr(self, "emolex_processors"):
            self.emolex_processors = {}
        if language not in self.emolex_processors:
            emolex_processor = KeywordProcessor()
            pbar = tqdm.tqdm(self.resource, desc="Indexing")
            for i, doc in enumerate(pbar):
                doc_lang = doc["language"].lower()
                if doc_lang != language:
                    continue
                emolex_processor.add_keyword(doc[self.keyword_field], i)
                # only perform lemmatization for supported languages
                # supported languages are:
                #   - English
                lemmatizer = None
                stemmer = None
                if doc_lang == "english":
                    lemmatizer = WordNetLemmatizer()
                    stemmer = SnowballStemmer("english")
                if lemmatizer is not None:
                    emolex_processor.add_keyword(
                        lemmatizer.lemmatize(doc[self.keyword_field]), i
                    )
                if stemmer is not None:
                    emolex_processor.add_keyword(
                        stemmer.stem(doc[self.keyword_field]), i
                    )
            pbar.close()
            self.emolex_processors[language] = emolex_processor
        return self.emolex_processors[language]

    def augment_with_lemmatizer(self, text: str, *, language: str = "english"):
        # just in case
        language = language.lower()
        emolex_processor = self.get_keyword_processor(language)
        if language != "english":
            return
        lemmatizer = WordNetLemmatizer()
        for token in word_tokenize(text):
            exists = emolex_processor.get_keyword(token)
            if exists is not None:
                continue
            lemma = lemmatizer.lemmatize(token)
            i = emolex_processor.get_keyword(lemma)
            if i is None:
                continue
            emolex_processor.add_keyword(token, int(i))
        stemmer = SnowballStemmer("english")
        for token in word_tokenize(text):
            exists = emolex_processor.get_keyword(token)
            if exists is not None:
                continue
            stem = stemmer.stem(token)
            i = emolex_processor.get_keyword(stem)
            if i is None:
                continue
            emolex_processor.add_keyword(token, int(i))

    def annotate(
        self,
        text: Union[str, List[str]],
        tokens: Union[Encoding, BatchEncoding] = None,
        return_tensors: Optional[str] = None,
        verbose: bool = False,
        language: str = "english",
    ) -> Union[List[dict], List[List[dict]], List[List[List[dict]]]]:
        language = language.lower()  # just in case, language must not be None
        if not isinstance(text, str):
            if verbose:
                text = tqdm.tqdm(text, desc="Annotating")
            annotations = [self.annotate(t, language=language) for t in text]
            if tokens is None:
                # List[List[dict]]
                return annotations
            if isinstance(tokens, BatchEncoding):
                # List[List[List[dict]]]
                return batch_align(tokens, annotations, self.labels, return_tensors)
            msg = f"unsupported tokens type: {tokens.__class__.__name__}"
            raise ValueError(msg)
        # augment the emolex processor with stemming
        self.augment_with_lemmatizer(text, language=language)
        kp = self.get_keyword_processor(language)
        annotations = []
        for ki, ks, ke in kp.extract_keywords(text, span_info=True):
            re_doc = self.resource[ki]
            if re_doc["language"].lower() != language:
                # check language
                continue
            for label_col in self.label_fields:
                for label, score in re_doc[label_col].items():
                    annotations.append({
                        "start": ks,
                        "end": ke,
                        "label": f"{self.namespace}/{label_col}/{label}",
                        "score": score,
                    })
        if tokens is None:
            # List[dict]
            return annotations
        if isinstance(tokens, Encoding):
            # List[List[dict]] if self.multilabel else List[dict]
            return align(tokens, annotations, self.labels, return_tensors)
        msg = f"unsupported tokens type: {tokens.__class__.__name__}"
        raise ValueError(msg)
