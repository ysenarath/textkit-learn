# python -m tklearn.utils.flashtext
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    Generic,
    Iterator,
    List,
    MutableMapping,
    Optional,
    Tuple,
    TypeVar,
    overload,
)

import unibreak

from tklearn.typing import UNDEFINED

T = TypeVar("T")


@dataclass
class TrieNode(Generic[T]):
    """Node class for the keyword trie."""

    value: Optional[T] = UNDEFINED
    children: Dict[str, TrieNode] = field(default_factory=dict)
    fail: Optional[TrieNode] = None
    depth: Optional[int] = None


class Lexicon(Generic[T], MutableMapping[str, T]):
    """A processor for efficiently finding and replacing keywords in text."""

    # these values should not be accessed directly
    _root: TrieNode[T]
    _len: int
    _updated: bool
    # case sensitive should not be updated after the initialization
    _case_sensitive: bool

    def __init__(self, case_sensitive: bool = False):
        self._root = TrieNode()
        self._len = 0
        self._updated = False
        self._case_sensitive = case_sensitive

    def tokeinze(self, texts: List[str], return_offsets: bool = False):
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

    def _get_node(self, key: str) -> Optional[TrieNode[T]]:
        trie = self._root
        for token in next(self.tokeinze([key])):
            if token not in trie.children:
                return None
            trie = trie.children[token]
        return trie

    def __iter__(self) -> Iterator[str]:
        yield from self.keys()

    def keys(self, prefix: str = "") -> Iterator[str]:
        """Return all keywords with a given prefix."""
        trie = self._get_node(prefix)
        if trie is None:
            return
        stack = [(prefix, trie)]
        while stack:
            prefix, trie = stack.pop()
            if trie.value is not UNDEFINED:
                yield prefix
            for key, child in trie.children.items():
                stack.append((f"{prefix} {key}", child))

    def __getitem__(self, key: str) -> T:
        node = self._get_node(key)
        if node is None or node.value is UNDEFINED:
            raise KeyError(key)
        return node.value

    def __contains__(self, key: str) -> bool:
        """Check if a keyword is present."""
        trie = self._get_node(key)
        if trie is None:
            return False
        return trie.value is not UNDEFINED

    def __len__(self) -> int:
        return self._len

    def __setitem__(self, key: str, value: T):
        """Add a keyword with a different clean word (replacement text)."""
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
        """Delete a keyword."""
        parents: List[Tuple[TrieNode[T], str]] = []
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
        """Build failure links using BFS."""
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
    def extract(self, text: str) -> Iterator[Tuple[T, int, int]]: ...
    @overload
    def extract(
        self, text: List[str]
    ) -> Iterator[Iterator[Tuple[T, int, int]]]: ...
    def extract(self, text: Any) -> Any:
        """Extract keywords from text."""
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
        if node is None:
            node = self._root
        for key, child in node.children.items():
            if child.value is UNDEFINED:
                print("  " * depth + key, "NULL")
            else:
                print("  " * depth + key, child.value)
            self.display(child, depth + 1)


class MatchIterator(Generic[T]):
    def __init__(
        self,
        trie: TrieNode[T],
        tokens: List[str],
        offsets: List[Tuple[int, int]],
    ):
        self.trie = trie
        self.tokens = tokens
        self.offsets = offsets
        self.idx = 0
        self.size = len(self.tokens)

    def next(self) -> Optional[Tuple[T, int, int]]:
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

    def __next__(self) -> Tuple[T, int, int]:
        result = self.next()
        while result is None:
            result = self.next()
        return result

    def __iter__(self) -> MatchIterator[T]:
        return self

    def __len__(self):
        return len(self.tokens)
