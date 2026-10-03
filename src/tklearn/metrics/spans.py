from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import ClassVar, Literal

import numpy as np

from tklearn.metrics._utils import average_scores, check_option, prf_score
from tklearn.metrics.base import Metric

__all__ = [
    "SpanF1",
    "SpanPrecision",
    "SpanRecall",
    "get_spans",
]

Average = Literal["micro", "macro", "weighted"]


def _ends(prev_tag: str, tag: str, prev_type: str, type_: str) -> bool:
    return (
        prev_tag in ("E", "S")
        or (prev_tag in ("B", "I") and tag in ("B", "S", "O"))
        or (prev_tag not in ("O", ".") and prev_type != type_)
    )


def _starts(prev_tag: str, tag: str, prev_type: str, type_: str) -> bool:
    return (
        tag in ("B", "S")
        or (prev_tag in ("E", "S", "O") and tag in ("E", "I"))
        or (tag not in ("O", ".") and prev_type != type_)
    )


def get_spans(
    tags: Sequence[str], suffix: bool = False
) -> list[tuple[str, int, int]]:
    """Extract labelled spans from a sequence of tags.

    Tags follow the IOB1, IOB2 (BIO), IOE or IOBES schemes, e.g. ``B-PER``,
    and are read leniently like conlleval and seqeval's default mode: an
    ``I-`` tag after ``O`` or after another type starts a new span.

    Parameters
    ----------
    tags : sequence of str
        One tag per token.
    suffix : bool, default=False
        Tags put the position last, e.g. ``PER-B``.

    Returns
    -------
    list of (str, int, int)
        Span type, first token and last token (inclusive).

    Examples
    --------
    >>> get_spans(["B-PER", "I-PER", "O", "B-LOC"])
    [('PER', 0, 1), ('LOC', 3, 3)]
    """
    spans = []
    prev_tag, prev_type, start = "O", "", 0
    for i, chunk in enumerate([*tags, "O"]):
        if suffix:
            tag, type_ = chunk[-1], chunk[:-1].rsplit("-", maxsplit=1)[0]
        else:
            tag, type_ = chunk[0], chunk[1:].split("-", maxsplit=1)[-1]
        type_ = type_ or "_"
        if _ends(prev_tag, tag, prev_type, type_):
            spans.append((prev_type, start, i - 1))
        if _starts(prev_tag, tag, prev_type, type_):
            start = i
        prev_tag, prev_type = tag, type_
    return spans


class _SpanMetric(Metric):
    _kind: ClassVar[str]

    def __init__(
        self,
        *,
        average: Average | None = "micro",
        zero_division: float = 0.0,
        suffix: bool = False,
    ) -> None:
        check_option("average", average, ("micro", "macro", "weighted", None))
        super().__init__()
        self.average = average
        self.zero_division = zero_division
        self.suffix = suffix
        self.add_state("tp", Counter())
        self.add_state("n_true", Counter())
        self.add_state("n_pred", Counter())
        self.add_state("n_sequences", 0)

    def update(
        self,
        y_true: Sequence[Sequence[str]],
        y_pred: Sequence[Sequence[str]],
    ) -> None:
        if len(y_true) and isinstance(y_true[0], str):
            msg = "y_true and y_pred must be sequences of tag sequences"
            raise TypeError(msg)
        if len(y_true) != len(y_pred):
            msg = (
                f"got {len(y_true)} true and {len(y_pred)} predicted sequences"
            )
            raise ValueError(msg)
        for i, (true_tags, pred_tags) in enumerate(zip(y_true, y_pred)):
            if len(true_tags) != len(pred_tags):
                msg = (
                    f"sequence {i} has {len(true_tags)} true and "
                    f"{len(pred_tags)} predicted tags"
                )
                raise ValueError(msg)
            true = set(get_spans(true_tags, self.suffix))
            pred = set(get_spans(pred_tags, self.suffix))
            self.n_true.update(type_ for type_, _, _ in true)
            self.n_pred.update(type_ for type_, _, _ in pred)
            self.tp.update(type_ for type_, _, _ in true & pred)
        self.n_sequences += len(y_true)

    def compute(self) -> float | dict[str, float]:
        if not self.n_sequences:
            msg = f"{type(self).__name__} has no samples; call update first"
            raise ValueError(msg)
        types = sorted(self.n_true.keys() | self.n_pred.keys())
        tp = np.array([self.tp[t] for t in types], dtype=np.int64)
        fp = np.array([self.n_pred[t] for t in types], dtype=np.int64) - tp
        fn = np.array([self.n_true[t] for t in types], dtype=np.int64) - tp
        if self.average is None:
            scores = prf_score(self._kind, tp, fp, fn, self.zero_division)
            return dict(zip(types, scores.tolist()))
        return average_scores(
            self._kind, tp, fp, fn, self.average, self.zero_division
        )


class SpanPrecision(_SpanMetric):
    """Precision of predicted spans, e.g. named entities.

    A predicted span is correct when a true span has the same type, start
    and end. Spans are read from tags with `get_spans`, which matches
    seqeval's default mode.

    Reads ``y_true`` and ``y_pred``: sequences of tag sequences, e.g.
    ``[["B-PER", "I-PER", "O"], ...]``.

    Parameters
    ----------
    average : {"micro", "macro", "weighted"} or None, default="micro"
        ``"micro"`` pools the counts of all span types; ``"macro"`` and
        ``"weighted"`` average the per-type scores, unweighted or weighted
        by the number of true spans. None returns a dict of per-type scores.
    zero_division : float, default=0.0
        Score when the denominator is 0.
    suffix : bool, default=False
        Tags put the position last, e.g. ``PER-B``.
    """

    _kind = "precision"


class SpanRecall(_SpanMetric):
    """Recall of true spans. Inputs and parameters are as in `SpanPrecision`."""

    _kind = "recall"


class SpanF1(_SpanMetric):
    """F1 score of spans. Inputs and parameters are as in `SpanPrecision`."""

    _kind = "fbeta"
