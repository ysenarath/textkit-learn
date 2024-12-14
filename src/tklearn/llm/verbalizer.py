from __future__ import annotations

import re
from typing import Iterable, Mapping, Optional

from typing_extensions import Protocol, runtime_checkable

__all__ = [
    "Verbalizer",
    "MulticlassVerbalizer",
    "MultilabelVerbalizer",
]


@runtime_checkable
class Verbalizer(Protocol):
    def verbalize(self, input: str) -> str:
        raise NotImplementedError


class MulticlassVerbalizer:
    classes: set[str]
    labels_re: re.Pattern

    def __init__(self, classes: Mapping[str, str]):
        self.classes = set(classes)
        classes = "|".join([
            f"(?P<{re.escape(key)}>{token})" for key, token in classes.items()
        ])
        self.labels_re = re.compile(classes, re.IGNORECASE)

    def verbalize(self, input: str) -> Optional[str]:
        m = self.labels_re.search(input)
        if m is None:
            return None
        for label in self.classes:
            if m.group(label) is not None:
                return label
        return None


class MultilabelVerbalizer:
    classes: set[str]
    labels_re: re.Pattern

    def __init__(self, classes: Iterable[str] | Mapping[str, str]):
        self.classes = set(classes)
        if not isinstance(classes, Mapping):
            classes = {label: label for label in classes}
        classes = "|".join([
            f"(?P<{re.escape(key)}>{token})" for key, token in classes.items()
        ])
        self.labels_re = re.compile(classes, re.IGNORECASE)

    def verbalize(self, input: str) -> list[str]:
        pos = 0
        m = self.labels_re.search(input, pos)
        labels = set()
        while m is not None:
            for label in self.classes:
                if m.group(label) is not None:
                    labels.add(label)
            m = self.labels_re.search(input, m.end())
        return list(labels)
