"""
This module constructs URIs for nodes (concepts) in various languages. This
puts the tools in conceptnet5.uri together with functions that normalize
terms and languages into a standard form (english_filter, simple_tokenize, LCODE_ALIASES).
"""

import re

from wordfreq import simple_tokenize
from wordfreq.preprocess import preprocess_text

from tklearn.kb.conceptnet.uri import concept_uri

STOPWORDS = ["the", "a", "an"]

DROP_FIRST = ["to"]

LCODE_ALIASES = {
    # Pretend that various Chinese languages and variants are equivalent. This
    # is linguistically problematic, but it's also very helpful for aligning
    # them on terms where they actually are the same.
    #
    # This would mostly be a problem if ConceptNet was being used to *generate*
    # Chinese natural language text.
    "cmn": "zh",
    "yue": "zh",
    "zh_tw": "zh",
    "zh_cn": "zh",
    "zh-tw": "zh",
    "zh-cn": "zh",
    "zh-hant": "zh",
    "zh-hans": "zh",
    "nds-de": "nds",
    "nds-nl": "nds",
    # The Malay language can be represented specifically as 'zsm', or as the
    # more common language code 'ms'. 'ms' can also mean the macrolanguage of
    # all the Malay languages, which encompasses other languages such as
    # Indonesian.
    #
    # Prior to 5.8, we considered this a reason to unify Indonesian and Malay
    # under the language code 'ms'. However, in most corpus data, they are
    # distinguished as 'id' and 'ms', with 'ms' meaning the Malay language
    # _specifically_. So now we only merge 'zsm' into 'ms', and leave 'id' as
    # its own language.
    "zsm": "ms",
    # We had to make a decision here on Norwegian. Norwegian Bokmål ('nb') and
    # Nynorsk ('nn') have somewhat different vocabularies but are mutually
    # intelligible. Informal variants of Norwegian, especially when spoken,
    # don't really distinguish them. Some Wiktionary entries don't distinguish
    # them either. And the CLDR data puts them both in the same macrolanguage
    # of Norwegian ('no').
    #
    # The catch is, Bokmål and Danish are *more* mutually intelligible than
    # Bokmål and Nynorsk, so maybe they should be the same language too. But
    # Nynorsk and Danish are less mutually intelligible.
    #
    # There is no language code that includes both Danish and Nynorsk, so
    # it would probably be inappropriate to group them all together. We will
    # take the easy route of making the language boundaries correspond to the
    # national boundaries, and say that 'nn' and 'nb' are both kinds of 'no'.
    #
    # More information: http://languagelog.ldc.upenn.edu/nll/?p=9516
    "nn": "no",
    "nb": "no",
    # Our sources have entries in Croatian, entries in Serbian, and entries
    # in Serbo-Croatian. Some of the Serbian and Serbo-Croatian entries
    # are written in Cyrillic letters, while all Croatian entries are written
    # in Latin letters. Bosnian and Montenegrin are in there somewhere,
    # too.
    #
    # Applying the same principle as Chinese, we will unify the language codes
    # into the macrolanguage 'sh' without unifying the scripts.
    "bs": "sh",
    "hr": "sh",
    "sr": "sh",
    "hbs": "sh",
    "sr-latn": "sh",
    "sr-cyrl": "sh",
    # More language codes that we would rather group into a broader language:
    "arb": "ar",  # Modern Standard Arabic -> Arabic
    "arz": "ar",  # Egyptian Arabic -> Arabic
    "ary": "ar",  # Moroccan Arabic -> Arabic
    "ckb": "ku",  # Central Kurdish -> Kurdish
    "mvf": "mn",  # Peripheral Mongolian -> Mongolian
    "tl": "fil",  # Tagalog -> Filipino
    "vro": "et",  # Võro -> Estonian
    "sgs": "lt",  # Samogitian -> Lithuanian
    "ciw": "oj",  # Chippewa -> Ojibwe
    "xal": "xwo",  # Kalmyk -> Oirat
    "ffm": "ff",  # Maasina Fulfulde -> Fula
}


def english_filter(tokens: list[str]) -> list[str]:
    """Given a list of tokens, remove a small list of English stopwords.

    Parameters
    ----------
    tokens : list[str]
        The list of tokens to filter.

    Returns
    -------
    list[str]
        The list of tokens with the stopwords removed.
    """
    non_stopwords = [token for token in tokens if token not in STOPWORDS]
    while non_stopwords and non_stopwords[0] in DROP_FIRST:
        non_stopwords = non_stopwords[1:]
    if non_stopwords:
        return non_stopwords
    else:
        return tokens


def preprocess_and_tokenize_text(lang: str, text: str) -> str:
    """
    Get a string made from the tokens in the text, joined by
    underscores.

    Parameters
    ----------
    lang : str
        The language of the text.
    text : str
        The text to preprocess and tokenize.

    Returns
    -------
    str
        The text after preprocessing and tokenization.
    """
    text = preprocess_text(text.replace("_", " "), lang)
    tokens = simple_tokenize(text)
    return "_".join(tokens)


def topic_to_concept(language: str, topic: str) -> str:
    """
    Get a canonical representation of a Wikipedia topic, which may include
    a disambiguation string in parentheses. Returns a concept URI that
    may be disambiguated as a noun.

    >>> topic_to_concept('en', 'Township (United States)')
    '/c/en/township/n/wp/united_states'
    """
    # find titles of the form Foo (bar)
    topic = topic.replace("_", " ")
    match = re.match(r"([^(]+) \(([^)]+)\)", topic)
    if not match:
        return standardized_concept_uri(language, topic)
    else:
        return standardized_concept_uri(
            language, match.group(1), "n", "wp", match.group(2)
        )


def standardized_concept_name(lang, text):
    raise NotImplementedError(
        "standardized_concept_name has been removed. "
        "Use preprocess_and_tokenize_text instead."
    )


def standardized_concept_uri(lang, text, *more):
    """
    Make the appropriate URI for a concept in a particular language, including
    removing English stopwords, normalizing the text in a way appropriate
    to that language (using the text normalization from wordfreq), and joining
    its tokens with underscores in a concept URI.

    This text normalization can smooth over some writing differences: for
    example, it removes vowel points from Arabic words, and it transliterates
    Serbian written in the Cyrillic alphabet to the Latin alphabet so that it
    can match other words written in Latin letters.

    'more' contains information to distinguish word senses, such as a part
    of speech or a WordNet domain. The items in 'more' get lowercased and
    joined with underscores, but skip many of the other steps -- for example,
    they won't have stopwords removed.

    >>> standardized_concept_uri('en', 'this is a test')
    '/c/en/this_is_test'
    >>> standardized_concept_uri('en', 'this is a test', 'n', 'example phrase')
    '/c/en/this_is_test/n/example_phrase'
    >>> standardized_concept_uri('sh', 'симетрија')
    '/c/sh/simetrija'
    """
    lang = lang.lower()
    if lang in LCODE_ALIASES:
        lang = LCODE_ALIASES[lang]
    if lang == "en":
        token_filter = english_filter
    else:
        token_filter = None
    text = preprocess_text(text.replace("_", " "), lang)
    tokens = simple_tokenize(text)
    if token_filter is not None:
        tokens = token_filter(tokens)
    norm_text = "_".join(tokens)
    more_text = []
    for item in more:
        if item is not None:
            tokens = simple_tokenize(item.replace("_", " "))
            if token_filter is not None:
                tokens = token_filter(tokens)
            more_text.append("_".join(tokens))
    return concept_uri(lang, norm_text, *more_text)


def valid_concept_name(text: str) -> bool:
    """
    Returns whether this text can be reasonably represented in a concept
    URI. This helps to protect against making useless concepts out of
    empty strings or punctuation.

    >>> valid_concept_name('word')
    True
    >>> valid_concept_name('the')
    True
    >>> valid_concept_name(',,')
    False
    >>> valid_concept_name(',')
    False
    >>> valid_concept_name('/')
    False
    >>> valid_concept_name(' ')
    False
    """
    tokens = simple_tokenize(text.replace("_", " "))
    return len(tokens) > 0
