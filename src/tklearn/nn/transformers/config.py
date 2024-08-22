from __future__ import annotations

from typing import Any

from transformers import (
    AutoConfig,
    BertConfig,
    DistilBertConfig,
    GPT2Config,
    PretrainedConfig,
    RobertaConfig,
)
from typing_extensions import Self

from tklearn.utils.targets import TargetType, type_of_target

__all__ = [
    "TransformerConfig",
]


class TransformerConfig:
    def __init__(self, config: PretrainedConfig | TransformerConfig) -> None:
        if isinstance(config, TransformerConfig):
            config = config._hf_config
        self._hf_config = config

    @classmethod
    def from_config(cls, config: PretrainedConfig | TransformerConfig) -> Self:
        return cls(config)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs: Any) -> Self:
        cfg = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        self = cls(cfg)
        if kwargs:
            self.update(**kwargs)
        return self

    def update(self, **kwargs: Any) -> None:
        if "target_type" in kwargs:
            # "regression", "single_label_classification", "multi_label_classification"
            target_type = kwargs["target_type"]
            if isinstance(target_type, TargetType):
                target_type = target_type.label
            if target_type == "continuous":
                kwargs["problem_type"] = "regression"
            elif target_type in {"binary", "multiclass"}:
                kwargs["problem_type"] = "single_label_classification"
            elif target_type == "multilabel-indicator":
                kwargs["problem_type"] = "multi_label_classification"
            else:
                msg = f"target type '{target_type}' not supported"
                raise ValueError(msg)
        if isinstance(self._hf_config, GPT2Config):
            cfg = {}
            for k, v in kwargs.items():
                if k == "hidden_size":
                    cfg["n_embd"] = v
                elif k == "output_dropout":
                    cfg["seq_classif_dropout"] = v
                else:
                    cfg[k] = v
            self._hf_config.update(cfg)
        elif isinstance(self._hf_config, (BertConfig, RobertaConfig)):
            if "output_dropout" in kwargs:
                kwargs["classifier_dropout"] = kwargs.pop("output_dropout")
            self._hf_config.update(kwargs)
        elif isinstance(self._hf_config, DistilBertConfig):
            cfg = {}
            for k, v in kwargs.items():
                if k == "hidden_size":
                    cfg["dim"] = v
                elif k == "output_dropout":
                    cfg["seq_classif_dropout"] = v
                else:
                    cfg[k] = v
            self._hf_config.update(cfg)
        else:
            msg = f"config type '{self._hf_config.__class__.__name__}' not supported"
            raise ValueError(msg)

    @property
    def hidden_size(self) -> int:
        if isinstance(self._hf_config, GPT2Config):
            return self._hf_config.n_embd
        elif isinstance(self._hf_config, (BertConfig, RobertaConfig)):  # BertConfig
            return self._hf_config.hidden_size
        elif isinstance(self._hf_config, DistilBertConfig):
            return self._hf_config.dim
        msg = f"config type '{self._hf_config.__class__.__name__}' not supported"
        raise ValueError(msg)

    @property
    def output_dropout(self) -> float:
        if isinstance(self._hf_config, GPT2Config):
            return self._hf_config.seq_classif_dropout
        elif isinstance(self._hf_config, (BertConfig, RobertaConfig)):  # BertConfig
            return self._hf_config.classifier_dropout
        elif isinstance(self._hf_config, DistilBertConfig):
            return self._hf_config.seq_classif_dropout
        msg = f"config type '{self._hf_config.__class__.__name__}' not supported"
        raise ValueError(msg)

    @property
    def target_type(self) -> TargetType:
        if hasattr(self._hf_config, "target_type"):
            return type_of_target(self._hf_config.target_type)
        problem_type = None
        if isinstance(self._hf_config, PretrainedConfig):
            problem_type: str = self._hf_config.problem_type
        if problem_type == "regression":
            return type_of_target("continuous")
        elif problem_type == "single_label_classification":
            return type_of_target("multiclass")
        elif problem_type == "multi_label_classification":
            return type_of_target("multilabel-indicator")
        msg = f"problem type '{problem_type}' not supported"
        raise ValueError(msg)

    @property
    def num_labels(self) -> int:
        if isinstance(self._hf_config, PretrainedConfig):
            return self._hf_config.num_labels
        msg = f"config type '{self._hf_config.__class__.__name__}' not supported"
        raise ValueError(msg)

    @num_labels.setter
    def num_labels(self, value: int) -> None:
        self.update(num_labels=value)

    @property
    def output_attentions(self) -> bool:
        if isinstance(self._hf_config, PretrainedConfig):
            return self._hf_config.output_attentions
        msg = f"config type '{self._hf_config.__class__.__name__}' not supported"
        raise ValueError(msg)

    @property
    def output_hidden_states(self) -> bool:
        if isinstance(self._hf_config, PretrainedConfig):
            return self._hf_config.output_hidden_states
        msg = f"config type '{self._hf_config.__class__.__name__}' not supported"
        raise ValueError(msg)

    @property
    def pooling_method(self) -> str:
        pooling_method = getattr(self._hf_config, "pooling_method", None)
        if pooling_method is not None:
            return pooling_method
        if isinstance(self._hf_config, GPT2Config):
            # last token in the sequence (by default this is "cls_index")
            return "last"
        elif isinstance(self._hf_config, (BertConfig, RobertaConfig, DistilBertConfig)):
            # [CLS] token
            return "first"
        msg = f"config type '{self._hf_config.__class__.__name__}' not supported"
        raise ValueError(msg)
