"""Lexicon class for keyword extraction and replacement.

This module provides a Lexicon class that uses a trie data structure
to efficiently find and replace keywords in text. The Lexicon class
supports case sensitivity, tokenization, and the ability to build
failure links for efficient matching. It also provides methods for
adding, deleting, and extracting keywords, as well as displaying the
trie structure.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator, MutableMapping
from dataclasses import dataclass, field
from typing import Any, Generic, Optional, TypeVar, overload

import unibreak
from typing_extensions import Literal

from tklearn.utils.constants import UNDEFINED

T = TypeVar("T")


@dataclass
class TrieNode(Generic[T]):
    """A node in a trie data structure used for keyword matching.

    This class implements a node for a trie (prefix tree) that is used in the
    Aho-Corasick string matching algorithm. Each node can store a value and
    maintains references to child nodes, a failure link for the Aho-Corasick
    algorithm, and its depth in the trie.

    Parameters
    ----------
    value : Optional[T], default=UNDEFINED
        The value associated with this node. UNDEFINED indicates no value is stored.
    children : dict[str, TrieNode], default=empty dict
        Dictionary mapping characters to child nodes.
    fail : Optional[TrieNode], default=None
        Reference to the failure node for the Aho-Corasick algorithm.
    depth : Optional[int], default=None
        The depth of this node in the trie.

    Attributes
    ----------
    value : Optional[T]
        The value stored in this node, if any.
    children : dict[str, TrieNode]
        Dictionary of child nodes, where keys are characters and values are TrieNodes.
    fail : Optional[TrieNode]
        Failure link used in the Aho-Corasick algorithm for efficient pattern matching.
    depth : Optional[int]
        The depth of this node in the trie structure.

    Examples
    --------
    >>> root = TrieNode[str]()  # Create root node
    >>> child = TrieNode[str](value="example")  # Create child with value
    >>> root.children["a"] = child  # Add child node
    >>> child.fail = root  # Set failure link
    >>> child.depth = 1  # Set depth
    """

    value: Optional[T] = UNDEFINED
    children: dict[str, TrieNode] = field(default_factory=dict)
    fail: Optional[TrieNode] = None
    depth: Optional[int] = None


class MatchIterator(Generic[T]):
    """Iterator for finding longest matches in text using the Aho-Corasick algorithm.

    This class implements an iterator that finds the longest matching patterns in text
    using a trie data structure with failure links (Aho-Corasick algorithm). It processes
    tokens sequentially and returns matches along with their positions in the original text.

    Parameters
    ----------
    trie : TrieNode[T]
        The root node of the trie containing patterns to match.
    tokens : list[str]
        List of tokens to search through.
    offsets : list[tuple[int, int]]
        List of token offsets (start, end) positions in the original text.

    Notes
    -----
    The iterator implements the Aho-Corasick algorithm with modifications to find
    the longest matches. When multiple patterns match at a position, it returns
    the longest one. The algorithm uses failure links in the trie for efficient
    matching.

    Examples
    --------
    >>> # Create a trie with patterns
    >>> trie = TrieNode[str]()
    >>> # Add patterns to trie...
    >>> tokens = ["the", "quick", "brown", "fox"]
    >>> offsets = [(0, 3), (4, 9), (10, 15), (16, 19)]
    >>> iterator = MatchIterator(trie, tokens, offsets)
    >>> for value, start, end in iterator:
    ...     print(f"Found match: {value} at positions {start}-{end}")
    """

    def __init__(
        self,
        trie: TrieNode[T],
        tokens: list[str],
        offsets: list[tuple[int, int]],
    ):
        self.trie = trie
        self.tokens = tokens
        self.offsets = offsets
        self.idx = 0
        self.size = len(self.tokens)

    def next(self) -> Optional[tuple[T, int, int]]:
        """Get the next match in the text.

        Returns
        -------
        Optional[tuple[T, int, int]]
            A tuple containing (value, start, end) where:
            - value: The value associated with the matched pattern
            - start: Starting position in the original text
            - end: Ending position in the original text
            Returns None if no more matches are found.

        Notes
        -----
        This method implements the core matching logic, finding the longest
        possible match at the current position using the trie structure
        and failure links.
        """
        if self.idx >= self.size:
            raise StopIteration

        longest_sequence = None
        longest_sequence_length = 0
        first_longest_end = 0
        traversal_start_idx = self.idx
        current_idx = self.idx

        node = self.trie

        while current_idx < self.size:
            end_token_idx = current_idx
            token = self.tokens[current_idx]
            current_idx += 1

            if token in node.children:
                # token found!
                node = node.children[token]
                if node.value is UNDEFINED:
                    pass
                elif longest_sequence is None:
                    longest_sequence = (
                        node.value,
                        traversal_start_idx,
                        end_token_idx,
                    )
                    longest_sequence_length = (
                        end_token_idx - traversal_start_idx
                    )
                    first_longest_end = end_token_idx
                else:
                    sequence_length = end_token_idx - traversal_start_idx
                    if sequence_length > longest_sequence_length:
                        longest_sequence = (
                            node.value,
                            traversal_start_idx,
                            end_token_idx,
                        )
                        longest_sequence_length = sequence_length
            elif node.fail is not None and node.fail.depth > 0:
                if end_token_idx - node.fail.depth > first_longest_end:
                    break
                # shift the start to the next possible match
                node = node.fail
                current_idx = end_token_idx
                traversal_start_idx = current_idx - node.depth
                continue
            else:
                # token not found in the node
                break

        self.idx = traversal_start_idx + 1

        if longest_sequence is None:
            return

        self.idx = longest_sequence[2] + 1

        return (
            longest_sequence[0],
            self.offsets[longest_sequence[1]][0],
            self.offsets[longest_sequence[2]][1],
        )

    def __next__(self) -> tuple[T, int, int]:
        """Get the next match, implementing the iterator protocol.

        Returns
        -------
        tuple[T, int, int]
            A tuple containing (value, start, end) for the next match.
            - value: The value associated with the matched pattern
            - start: Starting position in the original text
            - end: Ending position in the original text

        Raises
        ------
        StopIteration
            When no more matches are found.
        """
        result = self.next()
        while result is None:
            result = self.next()
        return result

    def __iter__(self) -> MatchIterator[T]:
        """Make the class iterable.

        Returns
        -------
        MatchIterator[T]
            Returns self to implement the iterator protocol.
        """
        return self

    def __len__(self):
        """Get the total number of tokens in the input text.

        Returns
        -------
        int
            The total number of tokens in the input text that this iterator
            is searching through. This represents the size of the search space,
            not the number of matches that will be found.
        """
        return len(self.tokens)


class Lexicon(Generic[T], MutableMapping[str, T]):
    """A generic keyword processor using a trie data structure.

    This class implements a keyword processor that uses a trie (prefix tree) data structure
    for efficient keyword matching and extraction from text. It supports case-sensitive
    and case-insensitive matching, and implements the MutableMapping interface for
    dictionary-like operations.

    Parameters
    ----------
    case_sensitive : bool, default=False
        Whether to perform case-sensitive matching of keywords.

    Attributes
    ----------
    _root : TrieNode[T]
        The root node of the trie data structure.
    _len : int
        Number of keywords stored in the lexicon.
    _updated : bool
        Flag indicating if the trie needs rebuilding of failure links.
    _case_sensitive : bool
        Flag indicating if keyword matching is case-sensitive.

    Notes
    -----
    The Lexicon class uses the Aho-Corasick algorithm for efficient pattern matching,
    which is built upon a trie data structure with failure links. This allows for
    O(n + m + k) time complexity for pattern matching, where n is the length of the
    text, m is the total length of the patterns, and k is the number of matches.

    Examples
    --------
    >>> # Create a case-insensitive lexicon
    >>> lexicon = Lexicon[str]()
    >>> # Add some keywords with their associated values
    >>> lexicon["New York"] = "NY"
    >>> lexicon["Los Angeles"] = "LA"
    >>> # Extract keywords from text
    >>> text = "I love New York and Los Angeles!"
    >>> for value, start, end in lexicon.extract(text):
    ...     print(f"Found {value} at positions {start}-{end}")
    Found NY at positions 7-15
    Found LA at positions 20-30
    """

    # these values should not be accessed directly
    _root: TrieNode[T]
    _len: int
    _updated: bool
    # case sensitive should not be updated after the initialization
    _case_sensitive: bool

    def __init__(self, case_sensitive: bool = False):
        """Initialize a new Lexicon instance.

        Parameters
        ----------
        case_sensitive : bool, default=False
            Whether to perform case-sensitive matching of keywords.
        """
        self._root = TrieNode()
        self._len = 0
        self._updated = False
        self._case_sensitive = case_sensitive

    def tokeinze(self, texts: list[str], return_offsets: bool = False):
        """Tokenize input texts into words with optional position offsets.

        Parameters
        ----------
        texts : list[str]
            List of input texts to tokenize.
        return_offsets : bool, default=False
            Whether to return token position offsets in the original text.

        Yields
        ------
        Union[list[str], tuple[list[str], list[tuple[int, int]]]]
            If return_offsets is False, yields list of tokens for each text.
            If return_offsets is True, yields tuple of (tokens, offsets) where
            offsets is a list of (start, end) positions for each token.

        Notes
        -----
        Uses the unibreak library for Unicode-aware word segmentation.
        Handles case conversion based on the lexicon's case_sensitive setting.

        Examples
        --------
        >>> lexicon = Lexicon()
        >>> # Without offsets
        >>> list(lexicon.tokeinze(["Hello World!"]))
        [['hello', 'world']]
        >>> # With offsets
        >>> list(lexicon.tokeinze(["Hello World!"], return_offsets=True))
        [(['hello', 'world'], [(0, 5), (6, 11)])]
        """
        if isinstance(texts, str):
            raise TypeError("expected a list of strings, got a string")
        for text in texts:
            tokens = []
            offsets = []
            start, end = 0, 0
            # token may be a space here!
            for token in unibreak.split_words(text):
                start = end
                end += len(token)
                token = token.strip()
                # remove spaces
                if not token:
                    continue
                if not self._case_sensitive:
                    token = token.lower()
                tokens.append(token)
                offsets.append((start, end))
                start = end
            if end != len(text):
                # TODO: use regex to split the words
                raise
            if return_offsets:
                yield tokens, offsets
            else:
                yield tokens

    @overload
    def _get_node(
        self, key: str, return_tokens: Literal[True]
    ) -> tuple[TrieNode[T], list[str]]: ...
    @overload
    def _get_node(
        self, key: str, return_tokens: Literal[False]
    ) -> Optional[TrieNode[T]]: ...
    def _get_node(self, key: str, return_tokens: bool = False) -> Any:
        """Get the trie node corresponding to a key.

        Parameters
        ----------
        key : str
            The key to look up in the trie.
        return_tokens : bool, default=False
            Whether to return the tokenized form of the key along with the node.

        Returns
        -------
        Union[Optional[TrieNode[T]], tuple[TrieNode[T], list[str]]]
            If return_tokens is False, returns the node if found, None otherwise.
            If return_tokens is True, returns tuple of (node, tokens).

        Notes
        -----
        This is an internal method used by other operations to traverse the trie.
        The method tokenizes the key and follows the path in the trie.
        """
        trie = self._root
        tokens = next(self.tokeinze([key]))
        for token in tokens:
            if token not in trie.children:
                return None
            trie = trie.children[token]
        if return_tokens:
            return trie, tokens
        return trie

    def __iter__(self) -> Iterator[str]:
        yield from self.keys()

    def keys(self, prefix: str = "") -> Iterator[str]:
        """Return an iterator over all keywords with a given prefix.

        Parameters
        ----------
        prefix : str, default=""
            The prefix to filter keywords. If empty, returns all keywords.

        Returns
        -------
        Iterator[str]
            Iterator yielding all keywords that start with the given prefix.

        Examples
        --------
        >>> lexicon = Lexicon[str]()
        >>> lexicon["apple"] = "fruit"
        >>> lexicon["application"] = "software"
        >>> list(lexicon.keys("app"))
        ['apple', 'application']
        """
        trie, tokens = self._get_node(prefix, return_tokens=True)
        if trie is None:
            return
        prefix = "".join(tokens)
        stack = [(prefix, trie)]
        while stack:
            prefix, trie = stack.pop()
            if trie.value is not UNDEFINED:
                yield prefix
            for key, child in trie.children.items():
                if prefix:
                    stack.append((f"{prefix} {key}", child))
                else:
                    stack.append((f"{key}", child))

    def __getitem__(self, key: str) -> T:
        node = self._get_node(key)
        if node is None or node.value is UNDEFINED:
            raise KeyError(key)
        return node.value

    def __contains__(self, key: str) -> bool:
        """Check if a keyword exists in the lexicon.

        Parameters
        ----------
        key : str
            The keyword to check.

        Returns
        -------
        bool
            True if the keyword exists in the lexicon, False otherwise.

        Examples
        --------
        >>> lexicon = Lexicon[str]()
        >>> lexicon["python"] = "language"
        >>> "python" in lexicon
        True
        >>> "java" in lexicon
        False
        """
        trie = self._get_node(key)
        if trie is None:
            return False
        return trie.value is not UNDEFINED

    def __len__(self) -> int:
        return self._len

    def __setitem__(self, key: str, value: T):
        """Add or update a keyword with an associated value.

        Parameters
        ----------
        key : str
            The keyword to add or update.
        value : T
            The value to associate with the keyword.

        Raises
        ------
        ValueError
            If the key is not a string or is empty.

        Notes
        -----
        Updates the trie structure and marks it for rebuilding failure links.
        If the keyword already exists, its value is updated.

        Examples
        --------
        >>> lexicon = Lexicon[str]()
        >>> lexicon["python"] = "language"  # Add new keyword
        >>> lexicon["python"] = "snake"     # Update existing keyword
        """
        trie = self._root

        if not isinstance(key, str):
            raise ValueError("only strings are allowed as keys")

        # Split word into tokens (similar to split_word_bounds in Rust)
        tokens = next(self.tokeinze([key]))

        for token in tokens:
            if token not in trie.children:
                trie.children[token] = TrieNode()
            trie = trie.children[token]

        if trie == self._root:
            raise ValueError("only non-empty strings are allowed as keys")

        # Increment len only if the keyword isn't already there
        if trie.value is UNDEFINED:
            self._len += 1
        trie.value = value

        # mark that the trie is updated
        self._updated = True

    def __delitem__(self, key: str):
        """Remove a keyword from the lexicon.

        Parameters
        ----------
        key : str
            The keyword to remove.

        Raises
        ------
        KeyError
            If the keyword doesn't exist in the lexicon.

        Notes
        -----
        Removes the keyword and cleans up any unused nodes in the trie.
        Marks the trie for rebuilding failure links.

        Examples
        --------
        >>> lexicon = Lexicon[str]()
        >>> lexicon["python"] = "language"
        >>> del lexicon["python"]  # Remove keyword
        >>> "python" in lexicon
        False
        """
        parents: list[tuple[TrieNode[T], str]] = []
        trie = self._root

        # Split word into tokens (similar to split_word_bounds in Rust)
        tokens = next(self.tokeinze([key]))

        for token in tokens:
            if token not in trie.children:
                raise KeyError(key)
            parents.append((trie, token))
            trie = trie.children[token]

        # Decrement len only if the keyword is there
        if trie.value is UNDEFINED:
            raise KeyError(key)

        self._len -= 1
        trie.value = UNDEFINED

        # Delete the nodes that are no longer needed
        for parent, token in reversed(parents):
            child = parent.children[token]
            if child.value is not UNDEFINED or child.children:
                break
            # remove the useless child node
            del parent.children[token]

        # mark that the trie is updated
        self._updated = True

    def build(self, force: bool = False):
        """Build failure links in the trie using breadth-first search.

        Parameters
        ----------
        force : bool, default=False
            If True, rebuilds failure links even if the trie hasn't been modified.

        Notes
        -----
        This method implements the Aho-Corasick algorithm's preprocessing step,
        building failure links that enable efficient pattern matching.
        The failure links are automatically rebuilt when needed during extraction,
        so manual calls to this method are usually unnecessary.

        Examples
        --------
        >>> lexicon = Lexicon[str]()
        >>> lexicon["cat"] = "animal"
        >>> lexicon["catch"] = "verb"
        >>> lexicon.build()  # Build failure links
        """
        if not (self._updated or force):
            return

        queue = deque()
        root = self._root
        root.depth = 0

        # Handle depth 1 nodes
        for key, child in root.children.items():
            queue.append(child)
            child.fail = root
            child.depth = 1

        # BFS for remaining nodes
        while queue:
            current: TrieNode = queue.popleft()

            for key, child in current.children.items():
                queue.append(child)

                child.depth = current.depth + 1

                failure = current.fail
                # get the root of the faliures
                while failure is not None and key not in failure.children:
                    failure = failure.fail

                child.fail = failure.children[key] if failure else root

        self._updated = False

    @overload
    def extract(self, text: str) -> Iterator[tuple[T, int, int]]: ...
    @overload
    def extract(
        self, text: list[str]
    ) -> Iterator[Iterator[tuple[T, int, int]]]: ...
    def extract(self, text: Any) -> Any:
        """Extract keywords from text using the Aho-Corasick algorithm.

        Parameters
        ----------
        text : Union[str, list[str]]
            Input text or list of texts to search for keywords.

        Returns
        -------
        Union[Iterator[tuple[T, int, int]], Iterator[Iterator[tuple[T, int, int]]]]
            For single text: Iterator yielding (value, start, end) tuples.
            For list of texts: Iterator of iterators, each yielding (value, start, end) tuples.
            - value: The value associated with the matched keyword
            - start: Starting position in the text
            - end: Ending position in the text

        Notes
        -----
        Automatically rebuilds failure links if the trie has been modified.
        Returns the longest possible matches when multiple keywords overlap.

        Examples
        --------
        >>> lexicon = Lexicon[str]()
        >>> lexicon["cat"] = "animal"
        >>> lexicon["catches"] = "verb"
        >>> text = "The cat catches mice"
        >>> for value, start, end in lexicon.extract(text):
        ...     print(f"Found {value} at {start}-{end}: {text[start:end]}")
        Found animal at 4-7: cat
        Found verb at 8-15: catches
        """
        # build the failure links if needed
        self.build()
        if isinstance(text, str):
            tokens, offsets = next(self.tokeinze([text], return_offsets=True))
            return MatchIterator(self._root, tokens, offsets)
        return (
            MatchIterator(self._root, tokens, offsets)
            for tokens, offsets in self.tokeinze(text, return_offsets=True)
        )

    def display(self, node: Optional[TrieNode[T]] = None, depth: int = 0):
        """Display the trie structure for debugging purposes.

        Parameters
        ----------
        node : Optional[TrieNode[T]], default=None
            The node to start displaying from. If None, starts from root.
        depth : int, default=0
            Current depth in the trie for indentation.

        Notes
        -----
        Prints a text representation of the trie structure. Useful for
        debugging and understanding the internal structure of the lexicon.

        Examples
        --------
        >>> lexicon = Lexicon[str]()
        >>> lexicon["cat"] = "animal"
        >>> lexicon["catch"] = "verb"
        >>> lexicon.display()
        cat animal
        catch verb
        """
        if node is None:
            node = self._root
        for key, child in node.children.items():
            if child.value is UNDEFINED:
                print("  " * depth + key, "NULL")
            else:
                print("  " * depth + key, child.value)
            self.display(child, depth + 1)
