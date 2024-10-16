from __future__ import annotations

from dataclasses import field
from typing import ClassVar

from nightjar import AutoModule, BaseConfig, BaseModule

from tklearn.models.backbone import AutoBackbone, Backbone, BackboneConfig
from tklearn.models.backbone.base import Tokenizer
from tklearn.nn import Module

__all__ = [
    "AutoModel",
    "Model",
    "ModelConfig",
]


class ModelConfig(BaseConfig, dispatch=["type"]):
    type: ClassVar[str]
    backbone: BackboneConfig = field(default_factory=BackboneConfig)


class AutoModel(AutoModule):
    def __new__(cls, config: ModelConfig) -> Model:
        approach = super().__new__(cls, config)
        if not isinstance(approach, Model):
            msg = (
                f"expected {Model.__name__}, "
                f"got {approach.__class__.__name__}"
            )
            raise TypeError(msg)
        return approach


class Model(BaseModule, Module):
    config: ModelConfig
    backbone: Backbone

    def __post_init__(self) -> None:
        self.backbone = AutoBackbone(self.config.backbone)

    @property
    def tokenizer(self) -> Tokenizer:
        if self.backbone.tokenizer is None:
            raise AttributeError("'tokenizer' is not available")
        return self.backbone.tokenizer
