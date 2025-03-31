from __future__ import annotations

from collections.abc import Mapping
from dataclasses import field
from typing import ClassVar

from nightjar import AutoModule, BaseConfig, BaseModule

from tklearn.nn import Module
from tklearn.nn.models.backbone import (
    AutoBackbone,
    Backbone,
    BackboneConfig,
    Tokenizer,
)

__all__ = [
    "AutoModel",
    "Model",
    "ModelConfig",
]


class ModelConfig(BaseConfig, dispatch=["type"]):
    type: ClassVar[str]
    backbone: BackboneConfig = field(default_factory=BackboneConfig)


class AutoModel(AutoModule):
    def __new__(cls, config: ModelConfig | Mapping) -> Model:
        if not isinstance(config, ModelConfig):
            if isinstance(config, Mapping):
                config = ModelConfig.from_dict(config)
            else:
                raise TypeError(
                    f"expected {ModelConfig.__name__}, got {config.__class__.__name__}"
                )
        model = super().__new__(cls, config)
        if not isinstance(model, Model):
            msg = f"expected {Model.__name__}, got {model.__class__.__name__}"
            raise TypeError(msg)
        return model


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
