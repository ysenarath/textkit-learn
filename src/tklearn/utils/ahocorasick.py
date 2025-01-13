from __future__ import annotations

from collections import deque
from typing import Dict, List, Set, Tuple


class TrieNode:
    def __init__(self):
        self.children: Dict[str, TrieNode] = {}
        self.fail: TrieNode = None
        self.output: Set[str] = set()


class AhoCorasick:
    def __init__(self, patterns: List[str]):
        """Initialize the Aho-Corasick automaton with a list of patterns."""
        self.root = TrieNode()
        self.patterns = patterns
        self._build_trie()
        self._build_failure_links()

    def _build_trie(self) -> None:
        """Build the trie structure from the patterns."""
        for pattern in self.patterns:
            current = self.root
            for char in pattern:
                if char not in current.children:
                    current.children[char] = TrieNode()
                current = current.children[char]
            current.output.add(pattern)

    def _build_failure_links(self) -> None:
        """Build failure links using BFS."""
        queue = deque()

        # Handle depth 1 nodes
        for char, node in self.root.children.items():
            queue.append(node)
            node.fail = self.root

        # BFS for remaining nodes
        while queue:
            current: TrieNode = queue.popleft()

            for char, child in current.children.items():
                queue.append(child)
                failure = current.fail

                while failure is not None and char not in failure.children:
                    failure = failure.fail

                child.fail = failure.children[char] if failure else self.root
                child.output.update(child.fail.output)

    def search(self, text: str) -> List[Tuple[int, str]]:
        """
        Search for all pattern occurrences in the text.
        Returns list of (position, pattern) tuples.
        """
        current = self.root
        results = []

        for i, char in enumerate(text):
            while current is not None and char not in current.children:
                current = current.fail

            if current is None:
                current = self.root
                continue

            current = current.children[char]
            for pattern in current.output:
                results.append((i - len(pattern) + 1, pattern))

        return sorted(results)


# Example usage
if __name__ == "__main__":
    # Initialize with patterns
    patterns = ["he", "she", "his", "hers"]
    ac = AhoCorasick(patterns)

    # Search in text
    text = "she sells seashells by the seashore"
    matches = ac.search(text)

    # Print results
    print(f"Text: {text}")
    print("\nMatches found:")
    for pos, pattern in matches:
        print(f"Pattern '{pattern}' found at position {pos}")
        print(
            f"Context: {text[max(0, pos - 5) : pos]}[{text[pos : pos + len(pattern)]}]{text[pos + len(pattern) : pos + len(pattern) + 5]}"
        )
