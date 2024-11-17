from collections import defaultdict
from typing import Iterable, List, Optional

import tqdm

from tklearn.core.vocab import Vocab
from tklearn.kb.base import KnowledgeBase
from tklearn.utils.flashtext import KeywordProcessor

__all__ = [
    "KeywordAnnotator",
]


class KeywordAnnotator:
    def __init__(
        self,
        ignore_chars: Optional[Iterable[str]] = None,
        verbose: bool = True,
    ):
        self.kp = KeywordProcessor()
        if ignore_chars is None:
            ignore_chars = {"-", "_"}
        self.ignore_chars = set(ignore_chars)
        self.verbose = verbose
        self.matches = defaultdict(set)

    @classmethod
    def from_vocab(
        cls, vocab: Iterable[str] | Vocab | KnowledgeBase, verbose: bool = True
    ):
        if isinstance(vocab, KnowledgeBase):
            vocab = vocab.vocab.tokens
        elif isinstance(vocab, Vocab):
            vocab = vocab.tokens
        self = cls()
        pbar = tqdm.tqdm(vocab, desc="Adding keywords", disable=not verbose)
        for lbl in pbar:
            self.matches[lbl].add(lbl)
            self.kp.add_keyword(lbl)
            for char in self.ignore_chars:
                if char in lbl:
                    new_label = lbl.replace(char, " ")
                    self.matches[new_label].add(lbl)
                    self.kp.add_keyword(new_label)
        return self

    def annotate(
        self, texts: str | List[str]
    ) -> List[List[tuple[str, int, int]]]:
        if isinstance(texts, str):
            texts = [texts]
        annotations = []
        for text in texts:
            if not isinstance(text, str):
                raise ValueError(
                    "annotate input must be a string or a list of strings"
                )
            for char in self.ignore_chars:
                if char in text:
                    text = text.replace(char, " ")
            text_annotations = [
                {
                    "string": text[start:end],
                    "start": start,
                    "end": end,
                    "candidates": list(self.matches[lbl]),
                }
                for (lbl, start, end) in self.kp.extract_keywords(
                    text, span_info=True
                )
            ]
            annotations.append(text_annotations)
        return annotations
