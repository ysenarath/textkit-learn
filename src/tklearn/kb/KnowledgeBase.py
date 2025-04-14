import functools
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

import nltk
from nltk.corpus import stopwords

from tklearn.kb.lexicon import Lexicon
from tklearn.kb.models import Candidate, Mention, Span, Triple
from tklearn.kb.triple_store import TripleStore

nltk.download("stopwords", quiet=True)


@functools.lru_cache(maxsize=1)
def get_stopwords(language: str = "english") -> set[str]:
    return set(stopwords.words(language))


def build_lexicon() -> Lexicon[set[str]]:
    """Populate lexicon from LevelDB."""
    lexicon = Lexicon[set[str]]()
    data_size = int(self.metadata.get(b"size").decode())
    for key, _ in track(
        self._get_prefixed_db("data").iterator(),
        total=data_size,
        description="Populating Lexicon",
    ):
        key = codec.decode(key)
        key_form = key[0]
        key_word = key[1]
        if key_form.strip() == "":
            continue
        if key_form not in lexicon:
            lexicon[key_form] = set()
        lexicon[key_form].add(key_word)
    return lexicon


class KnowledgeBase:
    lexicon: Lexicon[set[str]]
    triples: TripleStore

    def __init__(self, triples: TripleStore):
        index_subject = {}
        for triple in self.triples.query():
            if triple.subject not in index_subject:
                index_subject[triple.subject] = {}
            if triple.predicate not in index_subject[triple.subject]:
                index_subject[triple.subject][triple.predicate] = set()
            index_subject[triple.subject][triple.predicate].add(triple.object)
        self.index_subject = index_subject

    def extract_candidates(self, word: str, k: int = 5) -> Iterable[Candidate]:
        """Get all words and senses for a given form."""
        # get all the senses of the word
        db = self.triples.prefixed_db(join_key(word))
        for key, value in db.iterator():
            word, gloss = codec.decode(key)
            if len(value) == 0:
                continue
            sense_id = int.from_bytes(value, byteorder="big")
            embedding = None
            yield Candidate(word, gloss, embedding, sense_id).bind(self)

    def extract_mentions(
        self,
        text: str,
        min_word_len: int = 3,
        stopwords: Iterable[str] | str = "english",
    ) -> Iterable[Mention]:
        """Analyze text and return a list of words and their senses."""
        stopwords = get_stopwords(stopwords)
        stopwords = set(stopwords)
        for words, start, end in self.lexicon.extract(text):
            candidates = []
            form = text[start:end]
            if len(form) < min_word_len:
                continue
            if form.lower() in stopwords:
                continue
            for word in words:
                for cc in self.extract_candidates(word):
                    if cc.word.lower() != form.lower():
                        continue
                    candidates.append(cc)
            yield Mention(form, Span(start, end), candidates)

    def augment(
        self, text: str, mentions=None, filter_func=None
    ) -> Iterable[dict[str, Any]]:
        """Augment text by replacing words with synonyms, hyponyms, and hypernyms."""
        filter_func = filter_func or (lambda x: True)
        yield {
            "text": text,
            "original": text,
            "span.start": 0,
            "span.end": 0,
            "relations": [],
            "support": 0,
        }
        if mentions is None:
            mentions = self.extract_mentions(text)
        for mention in mentions:
            start, end = mention.span
            prefix, suffix = text[:start], text[end:]
            aug_words = defaultdict(set)
            for candidate in mention.candidates:
                subject = (candidate.word, candidate.sense_id)
                for predicate in ["synonym", "hyponym", "instance"]:
                    objects = self.index_subject.get(subject, {}).get(
                        predicate, []
                    )
                    for object_ in objects:
                        rel = Triple(subject, predicate, object_)
                        if not filter_func(rel):
                            continue
                        aug_words[rel.object[0]].add(rel.to_tuple())
            for word, relations in aug_words.items():
                augmented_text = None
                augmented_text = prefix + word + suffix
                yield {
                    "text": augmented_text,
                    "original": text,
                    "span.start": start,
                    "span.end": end,
                    "relations": list(relations),
                    "support": len(relations),
                }
