from typing import (
    Callable,
    Dict,
    Literal,
)

from numpy.typing import ArrayLike
from sklearn.metrics._classification import _check_targets  # noqa: PLC2701

TargetType = Literal[
    "continuous",
    "continuous-multioutput",
    "binary",
    "multiclass",
    "multiclass-multioutput",
    "multilabel-indicator",
    "unknown",
]

check_targets: Callable[
    [ArrayLike, ArrayLike], tuple[str, ArrayLike, ArrayLike]
] = _check_targets

NUM_ARGS = 2

TARGET_TYPES: Dict[TargetType, int] = {
    "binary": 0,
    "multiclass": 1,
    "multilabel-indicator": 2,
    "multiclass-multioutput": 3,
    "continuous": 4,
    "continuous-multioutput": 5,
    "unknown": 6,
}


def resolve_target_type(*ps: str) -> None:
    if not ps:
        msg = "at least one target type must be provided"
        raise ValueError(msg)
    ps = [p for p in ps if p is not None]
    if len(ps) == 1:
        return ps[0]
    if len(ps) > NUM_ARGS:
        p = ps[0]
        for i in range(1, len(ps)):
            p = resolve_target_type(p, ps[i])
        return p
    # sort the types by index
    p, q = sorted(ps, key=lambda x: TARGET_TYPES[x])
    try:
        return {
            ("binary", "multilabel-indicator"): "unknown",
            ("binary", "multiclass-multioutput"): "unknown",
            ("multilabel-indicator", "continuous"): "continuous-multioutput",
            ("multiclass-multioutput", "continuous"): "continuous-multioutput",
        }[(p, q)]
    except KeyError:
        return q
