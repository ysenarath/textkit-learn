from __future__ import annotations

import string
from typing import Iterable, List, Tuple

from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import Lemma, Synset
from nltk.tokenize import word_tokenize


def get_wordnet_pos(pos_tag: str):
    """Convert Penn Treebank POS tags to WordNet POS tags"""
    if pos_tag.startswith("J"):
        return wn.ADJ
    elif pos_tag.startswith("V"):
        return wn.VERB
    elif pos_tag.startswith("N"):
        return wn.NOUN
    elif pos_tag.startswith("R"):
        return wn.ADV
    else:
        return None


def get_context_window(
    tokens: list[str], target_word: str, window_size: int = -1
) -> List[str]:
    """Get words around the target word within a fixed window size"""
    tokens = [token.lower() for token in tokens]
    if window_size < 0:
        window_size = len(tokens)
    try:
        target_idx = tokens.index(target_word.lower())
        start = max(0, target_idx - window_size)
        end = min(len(tokens), target_idx + window_size + 1)
        return tokens[start:end]
    except ValueError:
        return tokens


def get_related_words(synset: Synset, depth: int = 2) -> List[str]:
    related: List[str] = []
    # Add words from definition
    related.extend(word_tokenize(synset.definition()))
    # Add example sentences
    for example in synset.examples():
        related.extend(word_tokenize(example))
    # Add lemma names
    lemmas_: List[Lemma] = synset.lemmas()
    related.extend(sum([word_tokenize(lemma.name()) for lemma in lemmas_], []))
    # Add hypernyms and hyponyms up to specified depth
    if depth > 0:
        for hypernym in synset.hypernyms():
            related.extend(get_related_words(hypernym, depth - 1))
        for hyponym in synset.hyponyms():
            related.extend(get_related_words(hyponym, depth - 1))
    # Clean and normalize
    related = [
        word.lower()
        for word in related
        if word.lower() not in stopwords.words("english")
        and word not in string.punctuation
    ]
    return related


def compute_score_by_overlap(
    context_words: Iterable[str], sense_words: Iterable[str]
) -> int:
    return len(set(context_words).intersection(sense_words))


def lesk_wsd(
    tokens: List[str], target_word: str, pos: str | None = None
) -> Synset | None:
    # Get context window around target word
    context_words = get_context_window(tokens, target_word)
    # If POS is not provided, try to determine it
    if pos is None:
        tagged = pos_tag([target_word])[0][1]
        pos = get_wordnet_pos(tagged)
    # Get all possible synsets for the target word
    synsets: List[Synset]
    if pos:
        synsets = wn.synsets(target_word, pos)
    else:
        synsets = wn.synsets(target_word)
    if not synsets:
        return None
    # Find the synset with the highest overlap with context
    best_score = 0
    best_sense = None
    for sense in synsets:
        # Get related words for this sense
        sense_words = get_related_words(sense)
        # Compute overlap between context and sense words
        score = compute_score_by_overlap(context_words, sense_words)
        if score > best_score:
            best_score = score
            best_sense = sense
    return best_sense


def disambiguate_sentence(sentence: str) -> List[Tuple[str, Synset]]:
    """Disambiguate all content words in a sentence"""
    tokens = word_tokenize(sentence)
    tagged: List[Tuple[str, str]] = pos_tag(tokens)
    results = []
    for word, tag in tagged:
        pos = get_wordnet_pos(tag)
        if pos is None:
            continue
        sense = lesk_wsd(tokens, word, pos)
        if sense is not None:
            results.append((word, sense))
    return results


# Example usage
if __name__ == "__main__":
    test_sentences = [
        "I caught a bass in the lake",
        "The bass played a solo",
        "The bank is near the river",
        "I went to the bank to get money",
        "He cast his line into the pool",
        "The pool was filled with swimmers",
    ]

    for sentence in test_sentences:
        print(f"\nSentence: {sentence}")
        results = disambiguate_sentence(sentence)

        print("Disambiguated words:")
        for word, sense in results:
            print()
            print(f"Sentence: {sentence}")
            print(f"Word: {word}")
            print(f"Sense: {sense}")
            print(f"Definition: {sense.definition()}")
            if sense.examples():
                print(f"Examples: {sense.examples()}")
            hypernyms: List[Synset] = sense.hypernyms()
            if hypernyms:
                print("Hypernyms:", [h.name() for h in hypernyms])
