from __future__ import annotations

from typing import Any, Dict, Optional, Union, overload

import torch
from transformers import AutoModel
from typing_extensions import Literal, Protocol, Self, runtime_checkable

from tklearn.nn.base import Module
from tklearn.nn.prototypes.config import PrototypeConfig
from tklearn.nn.prototypes.helpers import CosineSimilarity
from tklearn.nn.prototypes.loss import BatchPrototypeLoss
from tklearn.nn.utils.preprocessing import preprocess_input, preprocess_target
from tklearn.utils.targets import TargetType

TRANSFORMERS_INPUTS = {
    # https://huggingface.co/docs/transformers/v4.42.0/en/model_doc/bert#transformers.BertModel
    # input_ids, attention_mask, token_type_ids, position_ids, head_mask
    # https://huggingface.co/docs/transformers/v4.42.0/en/model_doc/distilbert#transformers.DistilBertModel
    # input_ids, attention_mask, head_mask
    # https://huggingface.co/docs/transformers/v4.42.0/en/model_doc/roberta#transformers.RobertaModel
    # input_ids, attention_mask, token_type_ids, position_ids, head_mask
    # dont need to pass "labels" to the model
    "input_ids",  # Indices of input sequence tokens in the vocabulary.
    "attention_mask",  # Mask to avoid performing attention on padding token indices.
    "token_type_ids",  # Segment token indices to indicate first and second portions of the inputs.
    "position_ids",  # Indices of positions of each input sequence tokens in the position embeddings.
    "head_mask",  # Mask to nullify selected heads of the self-attention modules.
}


@runtime_checkable
class OutputWithPooling(Protocol):
    pooler_output: torch.Tensor

    @overload
    def __getitem__(self, key: Literal["pooler_output"]) -> torch.Tensor: ...


class PrototypeForSequenceClassification(Module):
    def __init__(self, config: PrototypeConfig) -> None:
        super().__init__()
        self.config = config
        self.backbone = AutoModel.from_config(self.config._hf_config)
        self.loss_func = BatchPrototypeLoss()
        self.prototypes: Optional[torch.Tensor] = None
        self.similarity = CosineSimilarity()

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        target_type: Union[str, TargetType] = "multiclass",
        **kwargs,
    ) -> Self:
        config = PrototypeConfig.from_pretrained(
            pretrained_model_name_or_path,
            target_type=target_type,
            **kwargs,
        )
        return cls(config)

    def predict_step(
        self, batch: Dict[str, torch.Tensor], **kwargs
    ) -> OutputWithPooling:
        kwargs = {}
        for k, v in batch.items():
            if k not in TRANSFORMERS_INPUTS:
                continue
            kwargs[k] = v
        return self.backbone(**kwargs)

    def compute_loss(self, batch: Any, output: OutputWithPooling, **kwargs):
        targets, pooler_output = batch["labels"], output["pooler_output"]
        # dynamic number of labels therefore do not pass num_labels
        y_true = preprocess_target(self.config.target_type, targets)
        targets = y_true.to(torch.long)
        return self.loss_func(pooler_output, targets)

    def compute_metric_inputs(self, batch: Any, output: OutputWithPooling):
        targets, pooler_output = batch["labels"], output["pooler_output"]
        logits = self.similarity(pooler_output, self.prototypes)
        target_type = self.config.target_type
        y_true = preprocess_target(target_type, targets)
        y_pred, y_score = preprocess_input(target_type, logits)
        return {
            "y_true": y_true,
            "y_pred": y_pred,
            "y_score": y_score,
            "embedding": pooler_output,
        }
