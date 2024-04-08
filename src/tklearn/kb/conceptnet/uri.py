"""
URIs are Unicode strings that represent the canonical name for any object in
ConceptNet. These can be used with the ConceptNet Web API, or referred to in a
Semantic Web application, by attaching the prefix:

    http://api.conceptnet.io

For example, the English concept "book" has the URI '/c/en/book'. This concept
can be referred to, or retrieved, using this complete URI:

    http://api.conceptnet.io/c/en/book
"""

from typing import List, Optional, Tuple
from urllib.parse import urlparse

from typing_extensions import TypedDict


def join_uri(*pieces: str) -> str:
    """
    Join together a sequence of URI pieces, each of which may be a string or a
    list of strings. The result is a URI with slashes between the pieces.

    Parameters
    ----------
    pieces : list of str or list of str
        The pieces of the URI to join. If a piece is a list, it will be joined
        with slashes.

    Returns
    -------
    str
        The joined URI.
    """
    return "/" + ("/".join([piece.strip("/") for piece in pieces]))


def concept_uri(lang: str, text: str, *more: str) -> str:
    """
    Builds a representation of a concept, which is a word or
    phrase of a particular language, which can participate in relations with
    other concepts, and may be linked to concepts in other languages.

    Every concept has an ISO language code and a text. It may also have a part
    of speech (pos), which is typically a single letter. If it does, it may
    have a disambiguation, a string that distinguishes it from other concepts
    with the same text.

    This function should be called as follows, where arguments after `text`
    are optional:

        concept_uri(lang, text, pos, disambiguation...)

    `text` and `disambiguation` should be strings that have already been run
    through `preprocess_and_tokenize_text`.

    This is a low-level interface. See `standardized_concept_uri` in nodes.py for
    a more generally applicable function that also deals with special
    per-language handling.

    >>> concept_uri('en', 'cat')
    '/c/en/cat'
    >>> concept_uri('en', 'cat', 'n')
    '/c/en/cat/n'
    >>> concept_uri('en', 'cat', 'n', 'feline')
    '/c/en/cat/n/feline'
    >>> concept_uri('en', 'this is wrong')
    Traceback (most recent call last):
        ...
    AssertionError: 'this is wrong' is not in normalized form
    """
    assert " " not in text, "%r is not in normalized form" % text
    if len(more) > 0:
        if len(more[0]) != 1:
            # We misparsed a part of speech; everything after the text is
            # probably junk
            more = []
        for dis1 in more[1:]:
            assert " " not in dis1, "%r is not in normalized form" % dis1

    return join_uri("/c", lang, text, *more)


def compound_uri(op: str, args: List[str]) -> str:
    """
    Some URIs represent a compound structure or operator built out of a number
    of arguments. Some examples are the '/and' and '/or' operators, which
    represent a conjunction or disjunction over two or more URIs, which may
    themselves be compound URIs; or the assertion structure, '/a', which takes
    a relation and two URIs as its arguments.

    This function takes the main 'operator', with the slash included, and an
    arbitrary number of arguments, and produces the URI that represents the
    entire compound structure.

    These structures contain square brackets as segments, which look like
    `/[/` and `/]/`, so that compound URIs can contain other compound URIs
    without ambiguity.

    >>> compound_uri('/nothing', [])
    '/nothing/[/]'
    >>> compound_uri('/a', ['/r/CapableOf', '/c/en/cat', '/c/en/sleep'])
    '/a/[/r/CapableOf/,/c/en/cat/,/c/en/sleep/]'
    """
    items = [op]
    first_item = True
    items.append("[")
    for arg in args:
        if first_item:
            first_item = False
        else:
            items.append(",")
        items.append(arg)
    items.append("]")
    return join_uri(*items)


def split_uri(uri: str) -> List[str]:
    """
    Get the slash-delimited pieces of a URI.

    >>> split_uri('/c/en/cat/n/animal')
    ['c', 'en', 'cat', 'n', 'animal']
    >>> split_uri('/')
    []
    """
    if not uri.startswith("/"):
        return [uri]
    uri2 = uri.lstrip("/")
    if not uri2:
        return []
    return uri2.split("/")


def uri_prefix(uri: str, max_pieces: int = 3) -> str:
    """
    Strip off components that might make a ConceptNet URI too detailed. Only
    the first `max_pieces` components will be kept.

    By default, `max_pieces` is 3, making this function useful for converting
    disambiguated concepts into their more general ambiguous forms.

    If the URI is actually a fully qualified URL, no components are removed.

    >>> uri_prefix('/c/en/cat/n/animal')
    '/c/en/cat'
    >>> uri_prefix('/c/en/cat/n')
    '/c/en/cat'
    >>> uri_prefix('/c/en/cat')
    '/c/en/cat'
    >>> uri_prefix('/c/en')
    '/c/en'
    >>> uri_prefix('/c/en/cat', 2)
    '/c/en'
    >>> uri_prefix('http://en.wikipedia.org/wiki/Example')
    'http://en.wikipedia.org/wiki/Example'
    """
    if is_absolute_url(uri):
        return uri
    pieces = split_uri(uri)[:max_pieces]
    return join_uri(*pieces)


def uri_prefixes(uri: str, min_pieces: int = 2) -> List[str]:
    """
    Get URIs that are prefixes of a given URI: that is, they begin with the
    same path components. By default, the prefix must have at least 2
    components.

    If the URI has sub-parts that are grouped by square brackets, then
    only complete sub-parts will be allowed in prefixes.

    >>> list(uri_prefixes('/c/en/cat/n/animal'))
    ['/c/en', '/c/en/cat', '/c/en/cat/n', '/c/en/cat/n/animal']
    >>> list(uri_prefixes('/test/[/group/one/]/[/group/two/]'))
    ['/test/[/group/one/]', '/test/[/group/one/]/[/group/two/]']
    >>> list(uri_prefixes('http://en.wikipedia.org/wiki/Example'))
    ['http://en.wikipedia.org/wiki/Example']
    """
    if is_absolute_url(uri):
        return [uri]
    pieces = []
    prefixes = []
    for piece in split_uri(uri):
        pieces.append(piece)
        if len(pieces) >= min_pieces:
            if pieces.count("[") == pieces.count("]"):
                prefixes.append(join_uri(*pieces))
    return prefixes


def parse_compound_uri(uri: str) -> Tuple[str, List[str]]:
    """
    Given a compound URI, extract its operator and its list of arguments.

    >>> parse_compound_uri('/nothing/[/]')
    ('/nothing', [])
    >>> parse_compound_uri('/a/[/r/CapableOf/,/c/en/cat/,/c/en/sleep/]')
    ('/a', ['/r/CapableOf', '/c/en/cat', '/c/en/sleep'])
    >>> parse_compound_uri('/or/[/and/[/s/one/,/s/two/]/,/and/[/s/three/,/s/four/]/]')
    ('/or', ['/and/[/s/one/,/s/two/]', '/and/[/s/three/,/s/four/]'])
    """
    pieces = split_uri(uri)
    if pieces[-1] != "]":
        raise ValueError("Compound URIs must end with /]")
    if "[" not in pieces:
        raise ValueError(
            "Compound URIs must contain /[/ at the beginning of the argument list"
        )
    list_start = pieces.index("[")
    op = join_uri(*pieces[:list_start])

    chunks = []
    current = []
    depth = 0

    # Split on commas, but not if they're within additional pairs of brackets.
    for piece in pieces[(list_start + 1) : -1]:
        if piece == "," and depth == 0:
            chunks.append("/" + ("/".join(current)).strip("/"))
            current = []
        else:
            current.append(piece)
            if piece == "[":
                depth += 1
            elif piece == "]":
                depth -= 1

    assert depth == 0, "Unmatched brackets in %r" % uri
    if current:
        chunks.append("/" + ("/".join(current)).strip("/"))
    return op, chunks


def parse_possible_compound_uri(op, uri: str) -> List[str]:
    """
    The AND and OR conjunctions can be expressed as compound URIs, but if they
    contain only one thing, they are returned as just that single URI, not a
    compound.

    This function returns the list of things in the compound URI if its operator
    matches `op`, or a list containing the URI itself if not.

    >>> parse_possible_compound_uri(
    ...    'or', '/or/[/and/[/s/one/,/s/two/]/,/and/[/s/three/,/s/four/]/]'
    ... )
    ['/and/[/s/one/,/s/two/]', '/and/[/s/three/,/s/four/]']
    >>> parse_possible_compound_uri('or', '/s/contributor/omcs/dev')
    ['/s/contributor/omcs/dev']
    """
    if uri.startswith("/" + op + "/"):
        return parse_compound_uri(uri)[1]
    else:
        return [uri]


def conjunction_uri(*sources: str) -> str:
    """
    Make a URI representing a conjunction of sources that work together to provide
    an assertion. The sources will be sorted in lexicographic order.

    >>> conjunction_uri('/s/contributor/omcs/dev')
    '/s/contributor/omcs/dev'

    >>> conjunction_uri('/s/rule/some_kind_of_parser', '/s/contributor/omcs/dev')
    '/and/[/s/contributor/omcs/dev/,/s/rule/some_kind_of_parser/]'
    """
    if len(sources) == 0:
        # Logically, a conjunction with 0 inputs represents 'True', a
        # proposition that cannot be denied. This could be useful as a
        # justification for, say, mathematical axioms, but when it comes to
        # ConceptNet, that kind of thing makes us uncomfortable and shouldn't
        # appear in the data.
        raise ValueError("Conjunctions of 0 things are not allowed")
    elif len(sources) == 1:
        return sources[0]
    else:
        return compound_uri("/and", sorted(set(sources)))


def assertion_uri(rel: str, start: str, end: str) -> str:
    """
    Make a URI for an assertion, as a compound URI of its relation, start node,
    and end node.

    >>> assertion_uri('/r/CapableOf', '/c/en/cat', '/c/en/sleep')
    '/a/[/r/CapableOf/,/c/en/cat/,/c/en/sleep/]'
    """
    assert rel.startswith("/r"), rel
    return compound_uri("/a", (rel, start, end))


def is_concept(uri: str) -> bool:
    """
    >>> is_concept('/c/sv/klänning')
    True
    >>> is_concept('/x/en/ly')
    False
    >>> is_concept('/a/[/r/Synonym/,/c/ro/funcția_beta/,/c/en/beta_function/]')
    False
    """
    return uri.startswith("/c/")


def is_relation(uri: str) -> bool:
    """
    >>> is_relation('/r/IsA')
    True
    >>> is_relation('/c/sv/klänning')
    False
    """
    return uri.startswith("/r/")


def is_term(uri: str) -> bool:
    """
    >>> is_term('/c/sv/kostym')
    True
    >>> is_term('/x/en/ify')
    True
    >>> is_term('/a/[/r/RelatedTo/,/c/en/cake/,/c/en/flavor/]')
    False
    """
    return uri.startswith("/c/") or uri.startswith("/x/")


def is_absolute_url(uri: str) -> bool:
    """
    We have URLs pointing to Creative Commons licenses, starting with 'cc:',
    which for Linked Data purposes are absolute URLs because they'll be resolved
    into full URLs.

    >>> is_absolute_url('http://fr.wiktionary.org/wiki/mįkká’e_uxpáðe')
    True
    >>> is_absolute_url('/c/fr/nouveau')
    False
    """
    return uri.startswith("http") or uri.startswith("cc:")


def get_uri_language(uri: str) -> str:
    """
    Extract the language from a concept URI. If the URI points to an assertion,
    get the language of its first concept.

    >>> get_uri_language('/a/[/r/RelatedTo/,/c/en/orchestra/,/c/en/symphony/]')
    'en'
    >>> get_uri_language('/c/pl/cześć')
    'pl'
    >>> get_uri_language('/x/en/able')
    'en'
    """
    if uri.startswith("/a/"):
        return get_uri_language(parse_possible_compound_uri("a", uri)[1])
    elif is_term(uri):
        return split_uri(uri)[1]
    else:
        return None


def uri_to_label(uri: str) -> str:
    """
    Convert a ConceptNet uri into a label to be used in nodes. This function
    replaces an underscore with a space, so while '/c/en/example' will be
    converted into 'example', '/c/en/canary_islands' will be converted into
    'canary islands'.

    >>> uri_to_label('/c/en/example')
    'example'
    >>> uri_to_label('/c/en/canary_islands')
    'canary islands'
    >>> uri_to_label('/c/en')
    ''
    >>> uri_to_label('/r/RelatedTo')
    'RelatedTo'
    >>> uri_to_label('http://wikidata.dbpedia.org/resource/Q89')
    'Q89'
    """
    if is_absolute_url(uri):
        return uri.split("/")[-1].replace("_", " ")
    if is_term(uri):
        uri = uri_prefix(uri)
    parts = split_uri(uri)
    if len(parts) < 3 and not is_relation(uri):
        return ""
    return parts[-1].replace("_", " ")


class Licenses:
    cc_attribution = "cc:by/4.0"
    cc_sharealike = "cc:by-sa/4.0"


ConceptNodeJSONLD = TypedDict(
    "ConceptNodeJSONLD",
    {
        "@id": str,
        "label": str,
        "language": str,
        "sense_label": str,
        "term": str,
        "site": str,
        "site_available": bool,
        "path": str,
        "@type": str,
    },
)


def to_json_ld(uri: str, label: Optional[str] = None) -> ConceptNodeJSONLD:
    """Convert a ConceptNet URI into a dictionary suitable for Linked Data."""
    if label is None:
        label = uri_to_label(uri)
    ld = {"@id": uri, "label": label}
    if is_term(uri):
        pieces = split_uri(uri)
        ld["language"] = get_uri_language(uri)
        # Get a reasonably-distinct sense label for the term.
        # Usually it will be the part of speech, but when we have fine-grained
        # information from Wikipedia or WordNet, it'll include the last
        # component as well.
        if len(pieces) > 3:
            ld["sense_label"] = pieces[3]
        if len(pieces) > 4 and pieces[4] in ("wp", "wn"):
            ld["sense_label"] += ", " + pieces[-1]
        ld["term"] = uri_prefix(uri)
        ld["@type"] = "Node"
    elif uri.startswith("http"):
        domain = urlparse(uri).netloc
        ld["site"] = domain
        ld["term"] = uri
        # OpenCyc is down and UMBEL doesn't host their vocabulary on the
        # Web. This property indicates whether you can follow a link
        # via HTTP and retrieve more information.
        ld["site_available"] = True
        if domain in {"sw.opencyc.org", "umbel.org", "wikidata.dbpedia.org"}:
            ld["site_available"] = False
        ld["path"] = urlparse(uri).path
        ld["@type"] = "Node"
    elif uri.startswith("/r/"):
        ld["@type"] = "Relation"
    return ld
