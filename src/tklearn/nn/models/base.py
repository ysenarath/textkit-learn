from __future__ import annotations

from tklearn.nn.base.module import Module
from tklearn.nn.models.backbone import (
    Backbone,
    Tokenizer,
    TransformerBackbone,
)
from tklearn.utils.registry import Registry

__all__ = [
    "MODELS",
    "BackboneModel",
]


class BackboneModel(Module):
    """A `Module` built on a `Backbone` encoder.

    Parameters
    ----------
    backbone : Backbone or str
        The encoder, or a Hugging Face model id/path as a shorthand for
        ``TransformerBackbone(backbone)``.
    """

    def __init__(self, backbone: Backbone | str) -> None:
        super().__init__()
        if isinstance(backbone, str):
            backbone = TransformerBackbone(backbone)
        if not isinstance(backbone, Backbone):
            msg = (
                f"expected a Backbone or model name, got "
                f"{type(backbone).__name__}"
            )
            raise TypeError(msg)
        self.backbone = backbone

    @property
    def hidden_size(self) -> int:
        return self.backbone.hidden_size

    @property
    def tokenizer(self) -> Tokenizer:
        if self.backbone.tokenizer is None:
            raise AttributeError("'tokenizer' is not available")
        return self.backbone.tokenizer


#: Models by name, e.g. ``MODELS.create("linear", "bert-base-uncased", num_labels=3)``.
MODELS: Registry[BackboneModel] = Registry("model")
