from typing import Literal

TargetType = Literal[
    "continuous",
    "continuous-multioutput",
    "binary",
    "multiclass",
    "multiclass-multioutput",
    "multilabel-indicator",
    "unknown",
]
