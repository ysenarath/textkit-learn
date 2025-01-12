from __future__ import annotations

import re
import string
from typing import Any, List, NamedTuple, Optional, Tuple

import spacy_alignments as tokenizations
from nltk import (
    NLTKWordTokenizer,
    TweetTokenizer,
    WordNetLemmatizer,
    pos_tag_sents,
    sent_tokenize,
)
from nltk.corpus import stopwords as sw
from nltk.corpus import wordnet
from nltk.corpus.reader.wordnet import Lemma, Synset

from tklearn.utils import hashing

ISO639 = {"english": "eng", "russian": "rus"}


class Span(NamedTuple):
    start: int
    end: int

    @property
    def slice(self) -> slice:
        return slice(self.start, self.end)


def get_spans(text: str, tokens: List[str]) -> List[Optional[Span]]:
    _, b2a = tokenizations.get_alignments(list(text), tokens)
    spans = []
    for i in range(len(tokens)):
        if not b2a[i]:
            spans.append(None)
        else:
            spans.append(Span(min(b2a[i]), max(b2a[i]) + 1))
    return spans


class Token(NamedTuple):
    id: str
    text: str
    clean: str
    pos: Optional[str]
    lemma: str
    is_stopword: bool
    sentence: int
    span: Optional[Span]
    language: Optional[str] = None


class TextProcessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.tokenizer = NLTKWordTokenizer()
        self.stopwords = {}
        self.stopwords["english"] = set(sw.words("english"))
        self.punctrans = str.maketrans("", "", string.punctuation)

    def is_stopword(self, word: str, language: Optional[str] = None) -> bool:
        if language is None:
            return any(
                word in self.stopwords.get(lang, set())
                for lang in self.stopwords
            )
        return word in self.stopwords.get(language, set())

    def sent_tokenize(
        self, text: str, language: Optional[str] = None
    ) -> List[str]:
        if language is None:
            # assumes English by default
            return sent_tokenize(text)
        return sent_tokenize(text, language=language)

    def tokenize(self, text: str, language: Optional[str] = None) -> List[str]:
        # any language supported by NLTK
        raw_tokens = self.tokenizer.tokenize(text)
        if ('"' in text) or ("''" in text):
            # Find double quotes and converted quotes
            matched = [m.group() for m in re.finditer(r"``|'{2}|\"", text)]
            # Replace converted quotes back to double quotes
            tokens = [
                matched.pop(0) if tok in ['"', "``", "''"] else tok
                for tok in raw_tokens
            ]
        else:
            tokens = raw_tokens
        return tokens

    def pos_tag_sents(
        self, sents: List[List[str]], language: Optional[str] = None
    ) -> List[List[Tuple[Any, Optional[str]]]]:
        lang = ISO639.get(language, None)
        # here language is ISO 639 code of the language,
        #   e.g. 'eng' for English, 'rus' for Russian
        if lang is None:
            return [[(token, None) for token in tokens] for tokens in sents]
        return pos_tag_sents(sents, lang=lang)

    def lemmatize(self, word: str, pos: str, language: str) -> str:
        if language != "english":
            return word
        # supports only English
        pos = pos.lower()
        if pos == "j":
            pos = "a"  # 'j' <--> 'a' reassignment
        if pos in ["r"]:  # For adverbs it's a bit different
            ss: Synset = wordnet.synset(word + ".r.1")
            lemmas: List[Lemma] = ss.lemmas()
            return lemmas[0].pertainyms()[0].name()
        elif pos in ["a", "s", "v"]:  # For adjectives and verbs
            return self.lemmatizer.lemmatize(word, pos=pos)
        else:
            return self.lemmatizer.lemmatize(word)

    def preprocess(self, text: str) -> str:
        s = text.lower()
        # remove all punctuations
        x = s.translate(self.punctrans)
        # remove all digits (by checking if each character is a digit)
        x = "".join([i for i in x if not i.isdigit()])
        if len(x) > 0:
            # keep the text as it is if it has no characters
            s = x
        # remove all consecative spaces and \
        #   strip leading and trailing spaces
        return " ".join(s.split())

    def process(
        self, text: str, language: Optional[str] = None
    ) -> List[Token]:
        text_hash = hashing.hash(text)
        sentences = self.sent_tokenize(text, language=language)
        sent_tokens, sent_tokens_lower = [], []
        for sent_id, sentence in enumerate(sentences):
            st = self.tokenize(sentence, language=language)
            sent_tokens.append(st)
            sent_tokens_lower.append(list(map(self.preprocess, st)))
        sent_pos_tags = self.pos_tag_sents(
            sent_tokens_lower, language=language
        )
        tagged_tokens = [
            (sent_id, token, clean, pos)
            for sent_id, (tokens, pos_tags) in enumerate(
                zip(sent_tokens, sent_pos_tags)
            )
            for token, (clean, pos) in zip(tokens, pos_tags)
        ]
        token_spans = get_spans(text, [t[1] for t in tagged_tokens])
        tokens = []
        for token_id, (sent_id, token, clean, pos) in enumerate(tagged_tokens):
            span = token_spans[token_id]  # NOTE span can be None
            lemma = self.lemmatize(clean, pos, language)
            tokens.append(
                Token(
                    f"{text_hash}#{token_id}",
                    token,
                    clean,
                    pos,
                    lemma=lemma,
                    is_stopword=self.is_stopword(lemma, language),
                    sentence=sent_id,
                    span=span,
                    language=language,
                )
            )
        return tokens


class TweetTextProcessor(TextProcessor):
    def __init__(
        self,
        preserve_case: bool = True,
        reduce_len: bool = False,
        strip_handles: bool = False,
        match_phone_numbers: bool = True,
    ):
        super().__init__()
        self.tokenizer = TweetTokenizer(
            preserve_case=preserve_case,
            reduce_len=reduce_len,
            strip_handles=strip_handles,
            match_phone_numbers=match_phone_numbers,
        )

    def tokenize(self, text: str, language: Optional[str] = None) -> List[str]:
        return self.tokenizer.tokenize(text)
