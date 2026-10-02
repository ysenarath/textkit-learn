"""Plotting helpers.

Submodules import heavy plotting libraries, so names are loaded on first use.
"""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tklearn.plotting.embeddings import embed2d, plot_embedding
    from tklearn.plotting.matrix import plot_dot_matrix
    from tklearn.plotting.token_tree import TokenTree, plot_token_tree

__all__ = [
    "TokenTree",
    "embed2d",
    "plot_dot_matrix",
    "plot_embedding",
    "plot_token_tree",
]

_LOCATIONS = {
    "TokenTree": "tklearn.plotting.token_tree",
    "embed2d": "tklearn.plotting.embeddings",
    "plot_dot_matrix": "tklearn.plotting.matrix",
    "plot_embedding": "tklearn.plotting.embeddings",
    "plot_token_tree": "tklearn.plotting.token_tree",
}


def __getattr__(name: str):
    if name in _LOCATIONS:
        return getattr(import_module(_LOCATIONS[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
