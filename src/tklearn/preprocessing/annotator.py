from collections import defaultdict
from typing import TYPE_CHECKING, Iterable, List

from textrush import KeywordProcessor
from tqdm import auto as tqdm

from tklearn import logging
from tklearn.core.vocab import Vocab, VocabItem
from tklearn.kb.base import KnowledgeBase

__all__ = [
    "KeywordAnnotator",
]

logger = logging.get_logger(__name__)


class KeywordAnnotator:
    def __init__(
        self,
        ignore_chars: str = "-_",
        case_sensitive: bool = False,
        verbose: bool = True,
    ):
        self.kp = KeywordProcessor(case_sensitive=case_sensitive)
        self.ignore_chars = set(ignore_chars) or set()
        self.verbose = verbose
        self.matches = defaultdict(set)

    @classmethod
    def from_vocab(
        cls,
        vocab: Iterable[str] | Vocab | KnowledgeBase,
        ignore_chars: str = "-_",
        case_sensitive: bool = False,
        verbose: bool = True,
    ):
        kwargs = {
            "ignore_chars": ignore_chars,
            "case_sensitive": case_sensitive,
            "verbose": verbose,
        }
        if isinstance(vocab, KnowledgeBase):
            vocab = vocab.get_vocab()
        self = cls(**kwargs)
        if not TYPE_CHECKING:
            vocab = tqdm.tqdm(
                vocab, desc="Adding keywords", disable=not verbose
            )
        for keyword in vocab:
            if isinstance(keyword, VocabItem):
                keyword = keyword.token
            self.matches[keyword].add(keyword)
            self._add_keyword_ignore_errors(keyword)
            alt = keyword
            for char in self.ignore_chars:
                alt = alt.replace(char, " ")
            alt = alt.strip()
            self.matches[alt].add(keyword)
            self._add_keyword_ignore_errors(alt)
        return self

    def _add_keyword_ignore_errors(self, keyword: str):
        try:
            self.kp.add_keyword(keyword)
        except Exception as e:
            if self.verbose:
                logger.error(f"Error adding keyword {keyword}: {e}")

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
                text = text.replace(char, " ")
            text_annotations = [
                {
                    "string": text[start:end],
                    "start": start,
                    "end": end,
                    "matches": list(self.matches[clean_name]),
                }
                for (clean_name, start, end) in self.kp.extract_keywords(
                    text, span_info=True
                )
            ]
            annotations.append(text_annotations)
        return annotations
