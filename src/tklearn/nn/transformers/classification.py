from os import PathLike
from typing import Dict, TypeVar, Union

import torch
from torch import nn
from transformers import (
    AutoModel,
    BertConfig,
    DistilBertConfig,
    PreTrainedModel,
    RobertaConfig,
)
from transformers.modeling_outputs import BaseModelOutput
from typing_extensions import Self

from tklearn.nn.loss import TargetBasedLoss
from tklearn.nn.module import Module
from tklearn.nn.transformers.base import TransformerConfig
from tklearn.nn.transformers.outputs import SequenceClassifierOutput
from tklearn.nn.utils.data import RecordBatch
from tklearn.nn.utils.preprocessing import preprocess_input, preprocess_target
from tklearn.utils.targets import TargetType

__all__ = [
    "TransformerForSequenceClassification",
]

CT = TypeVar("CT", BertConfig, DistilBertConfig, RobertaConfig)
OutputType = Union[SequenceClassifierOutput, Dict[str, torch.Tensor], BaseModelOutput]

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


class OutputLayer(nn.Module):
    """Head for sentence-level classification/regression tasks."""

    def __init__(self, num_labels: int, hidden_size: int, dropout: float = 0.2):
        super().__init__()
        # pre-classifier dense layer
        self.dense = nn.Linear(hidden_size, hidden_size)
        # dropout layer
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()
        # classifier layer
        self.output = nn.Linear(hidden_size, num_labels)

    def forward(self, pooler_output, **kwargs):
        # last_hidden_state shape: (batch_size, seq_len, hidden_size)
        # x = last_hidden_state[:, 0, :]  # take [CLS] or <s> token
        x = torch.tanh(self.dense(pooler_output))
        x = self.dropout(x)
        x = self.output(x)
        return x


class TransformerForSequenceClassification(Module):
    def __init__(
        self,
        pretrained_model_name_or_path: Union[str, PathLike[str]],
        *,
        num_labels: int = 2,
        output_dropout: float = 0.2,
        output_attentions: bool = True,
        output_hidden_states: bool = True,
        target_type: Union[str, TargetType] = "multiclass",
    ):
        super().__init__()
        # init base model
        self.config = TransformerConfig.from_pretrained(
            pretrained_model_name_or_path,
            num_labels=num_labels,
            output_dropout=output_dropout,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            target_type=target_type,
        )
        self.base_model: PreTrainedModel = AutoModel.from_config(self.config._hf_config)
        self.output_layer = OutputLayer(
            num_labels=num_labels,
            hidden_size=self.config.hidden_size,
            dropout=self.config.output_dropout,
        )
        self._loss_func = TargetBasedLoss(
            self.config.target_type, num_labels=num_labels
        )

    def forward(self, **kwargs):
        outputs = self.base_model(**kwargs)
        # pooler output is the representation of the [CLS]
        # token in BERT-like models
        if hasattr(outputs, "pooler_output"):
            pooler_output = outputs.pooler_output
        elif hasattr(outputs, "last_hidden_state"):
            pooler_output = outputs.last_hidden_state[:, 0, :]
        else:
            pooler_output = outputs.hidden_states[-1][:, 0, :]
        logits = self.output_layer(pooler_output)
        # results.logits /= self.temperature
        return SequenceClassifierOutput.from_output(
            outputs, logits=logits, pooler_output=pooler_output
        )

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: Union[str, PathLike[str]]
    ) -> Self:
        config = TransformerConfig.from_pretrained(pretrained_model_name_or_path)
        # init base model
        return cls(
            pretrained_model_name_or_path,
            num_labels=config.num_labels,
            output_dropout=config.output_dropout,
            output_attentions=config.output_attentions,
            output_hidden_states=config.output_hidden_states,
            target_type=config.target_type,
        )

    def predict_on_batch(self, batch: RecordBatch) -> OutputType:
        kwargs = {}
        for k, v in batch.x.items():
            if k not in TRANSFORMERS_INPUTS:
                continue
            kwargs[k] = v
        return self(**kwargs)

    def compute_loss(self, batch: RecordBatch, output: OutputType, **kwargs):
        logits = output["logits"]
        input = batch.y or batch.x["labels"]
        target_type, num_labels = self.config.target_type, self.config.num_labels
        y_true = preprocess_input(target_type, input, num_labels=num_labels)
        return self._loss_func(logits, y_true)

    def prepare_metric_inputs(self, batch: RecordBatch, output: OutputType):
        input = batch.y or batch.x["labels"]
        target_type, num_labels = self.config.target_type, self.config.num_labels
        y_true = preprocess_input(target_type, input, num_labels=num_labels)
        logits = output["logits"]
        y_pred, y_score = preprocess_target(target_type, logits)
        return {"y_true": y_true, "y_pred": y_pred, "y_score": y_score}
