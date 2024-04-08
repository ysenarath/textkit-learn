import re
from typing import Mapping, Union

from typing_extensions import Protocol, runtime_checkable

__all__ = [
    "Verbalizer",
    "RegexVerbalizer",
]


@runtime_checkable
class Verbalizer(Protocol):
    def verbalize(self, input: str) -> str:
        raise NotImplementedError


class RegexVerbalizer:
    labels_re: re.Pattern
    labels: set[str]
    multilabel: bool

    def __init__(self, labels: Mapping[str, str], multilabel: bool = False):
        self.labels = set(labels)
        labels = "|".join([
            f"(?P<{re.escape(key)}>{token})" for key, token in labels.items()
        ])
        self.labels_re = re.compile(labels, re.IGNORECASE)
        self.multilabel = multilabel

    def _verbalize_multilabel(self, input: str) -> list[str]:
        pos = 0
        m = self.labels_re.search(input, pos)
        labels = set()
        while m is not None:
            for label in self.labels:
                if m.group(label) is not None:
                    labels.add(label)
            m = self.labels_re.search(input, m.end())
        return list(labels)

    def _verbalize_multiclass(self, input: str) -> str:
        m = self.labels_re.search(input)
        if m is None:
            return None
        for label in self.labels:
            if m.group(label) is not None:
                return label
        return None

    def verbalize(self, input: str) -> Union[str, list[str]]:
        if self.multilabel:
            return self._verbalize_multilabel(input)
        return self._verbalize_multiclass(input)
