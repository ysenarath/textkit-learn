from __future__ import annotations

import abc
from typing import Any, Dict, Optional, Type, TypeVar

from sklearn.utils.multiclass import type_of_target as _type_of_target

__all__ = [
    "type_of_target",
    "TargetType",
    "ContinuousTargetType",
    "ContinuousMultioutputTargetType",
    "BinaryTargetType",
    "MulticlassTargetType",
    "MulticlassMultioutputTargetType",
    "MultilabelIndicatorTargetType",
    "UnknownTargetType",
]

T = TypeVar("T")

_dispatch: Dict[str, Type[TargetType]] = {}


class TargetType(abc.ABC):
    label: str
    description: Optional[str]

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        _dispatch[cls.label] = cls

    @abc.abstractmethod
    def is_multioutput(self) -> bool:
        pass

    @abc.abstractmethod
    def is_continuous(self) -> bool:
        pass

    def __repr__(self) -> str:
        return f'TargetType(label="{self.label}", description="{self.description}")'


class ContinuousTargetType(TargetType):
    label = "continuous"
    description = "Regression (1D continuous targets)"

    def is_multioutput(self) -> bool:
        return False

    def is_continuous(self) -> bool:
        return True


class ContinuousMultioutputTargetType(TargetType):
    label = "continuous-multioutput"
    description = "Multioutput Regression (2D continuous targets)"

    def is_multioutput(self) -> bool:
        return True

    def is_continuous(self) -> bool:
        return True


class BinaryTargetType(TargetType):
    label = "binary"
    description = "Binary Classification (1D binary targets)"

    def is_multioutput(self) -> bool:
        return False

    def is_continuous(self) -> bool:
        return False


class MulticlassTargetType(TargetType):
    label = "multiclass"
    description = "Multiclass Classification (>2 discrete classes, 1D)"

    def is_multioutput(self) -> bool:
        return False

    def is_continuous(self) -> bool:
        return False


class MulticlassMultioutputTargetType(TargetType):
    label = "multiclass-multioutput"
    description = "Multiclass-Multioutput (>2 discrete classes, 2D)"

    def is_multioutput(self) -> bool:
        return True

    def is_continuous(self) -> bool:
        return False


class MultilabelIndicatorTargetType(TargetType):
    label = "multilabel-indicator"
    description = "Multilabel Classification (2D binary indicator matrix)"

    def is_multioutput(self) -> bool:
        return True

    def is_continuous(self) -> bool:
        return False


class UnknownTargetType(TargetType):
    label = "unknown"
    description = "Unknown (possibly 3D array, sequence of sequences, or array of non-sequence objects)"

    def is_multioutput(self) -> bool:
        msg = "cannot determine if target is multioutput for unknown target type"
        raise ValueError(msg)

    def is_continuous(self) -> bool:
        msg = "cannot determine if target is continuous for unknown target type"
        raise ValueError(msg)


def type_of_target(y: Any) -> TargetType:
    if isinstance(y, TargetType):
        return y
    tt: str = y if isinstance(y, str) else _type_of_target(y)
    return _dispatch.get(tt, UnknownTargetType)()


TARGET_TYPE_RANK = {
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
    if len(ps) > 2:
        p = ps[0]
        for i in range(1, len(ps)):
            p = resolve_target_type(p, ps[i])
        return p
    p, q = sorted(ps, key=lambda x: TARGET_TYPE_RANK[x])
    try:
        return {
            ("binary", "multilabel-indicator"): "unknown",
            ("binary", "multiclass-multioutput"): "unknown",
            ("multilabel-indicator", "continuous"): "continuous-multioutput",
            ("multiclass-multioutput", "continuous"): "continuous-multioutput",
        }[(p, q)]
    except KeyError:
        return q
