from os import PathLike
from typing import Any, Dict, TypeVar, Union

import torch
from torch import nn
from transformers import (
    AutoModel,
    BertConfig,
    DistilBertConfig,
    RobertaConfig,
)
from transformers.modeling_outputs import BaseModelOutput

from tklearn.nn import Module
from tklearn.nn.loss import TargetBasedLoss
from tklearn.nn.transformers.config import TransformerConfig
from tklearn.nn.transformers.outputs import SequenceClassifierOutput
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


class ClassifierLayer(nn.Module):
    """Head for sentence-level classification/regression tasks."""

    def __init__(self, num_labels: int, hidden_size: int, dropout: float = 0.2):
        super().__init__()
        # pre-classifier dense layer
        # self.dense = nn.Linear(hidden_size, hidden_size)
        # dropout layer
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()
        # classifier layer
        self.output = nn.Linear(hidden_size, num_labels)

    def forward(self, pooler_output, **kwargs):
        # last_hidden_state shape: (batch_size, seq_len, hidden_size)
        # x = last_hidden_state[:, 0, :]  # take [CLS] or <s> token
        # x = self.dense(pooler_output)
        # x = torch.tanh(x)
        x = self.dropout(pooler_output)
        x = self.output(x)
        return x, pooler_output


class TransformerForSequenceClassification(Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.base_model = AutoModel.from_config(config._hf_config)
        self.classifier = ClassifierLayer(
            num_labels=config.num_labels,
            hidden_size=config.hidden_size,
            dropout=config.output_dropout,
        )
        self.loss_func = TargetBasedLoss(
            config.target_type, num_labels=config.num_labels
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, PathLike[str]],
        *,
        num_labels: int = 2,
        output_dropout: float = 0.2,
        output_attentions: bool = True,
        output_hidden_states: bool = True,
        target_type: Union[str, TargetType] = "multiclass",
    ):
        # init base model
        config = TransformerConfig.from_pretrained(
            pretrained_model_name_or_path,
            num_labels=num_labels,
            output_dropout=output_dropout,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            target_type=target_type,
        )
        return cls(config)

    def forward(self, **kwargs):
        outputs = self.base_model(**kwargs)
        # pooler output is the representation of
        # the [CLS] or it could be the average of
        # all tokens token in BERT-like models
        if hasattr(outputs, "pooler_output"):
            pooler_output = outputs.pooler_output
        else:
            last_hidden_state: torch.Tensor
            if hasattr(outputs, "last_hidden_state"):
                last_hidden_state = outputs.last_hidden_state
            else:
                last_hidden_state = outputs.hidden_states[-1]
            attention_mask: torch.Tensor = kwargs.get("attention_mask")
            if self.config.pooling_method == "mean":
                if attention_mask is None:
                    pooler_output = last_hidden_state.mean(dim=1)
                else:
                    # Create a mask for the non-padded tokens
                    mask_expanded = attention_mask.unsqueeze(-1).expand(
                        last_hidden_state.size()
                    )
                    # sum the hidden states of non-padded tokens
                    sum_hidden_state = torch.sum(
                        last_hidden_state * mask_expanded,
                        dim=1,  # sum over the sequence length
                    )
                    # sum the attention mask (sum over the sequence length)
                    sum_attention_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                    # calculate the mean of the hidden states
                    pooler_output = sum_hidden_state / sum_attention_mask
            if self.config.pooling_method == "last":
                if attention_mask is None:
                    pooler_output = last_hidden_state[:, -1, :]
                else:
                    batch_size, seq_length, _ = last_hidden_state.shape
                    range_tensor = (
                        torch.arange(seq_length, device=attention_mask.device)
                        .unsqueeze(0)
                        .expand(batch_size, -1)
                    )
                    token_idxs = range_tensor * attention_mask
                    # Create a mask for the last token positions
                    last_token_idx = token_idxs.max(axis=1).values
                    pooler_output = last_hidden_state[
                        torch.arange(
                            last_hidden_state.size(0), device=last_hidden_state.device
                        ),
                        last_token_idx,
                    ]
            elif self.config.pooling_method == "first":
                if attention_mask is None:
                    pooler_output = last_hidden_state[:, 0, :]
                else:
                    # Create a mask for the first token positions
                    first_token_idx = attention_mask.max(axis=1).indices
                    pooler_output = last_hidden_state[
                        torch.arange(
                            last_hidden_state.size(0), device=last_hidden_state.device
                        ),
                        first_token_idx,
                    ]
        logits, pooler_output = self.classifier(pooler_output)
        # results.logits /= self.temperature
        return SequenceClassifierOutput.from_output(
            outputs,
            logits=logits,
            pooler_output=pooler_output,
        )

    def compute_loss(self, batch: Any, output: OutputType, **kwargs):
        targets, logits = batch["labels"], output["logits"]
        target_type, num_labels = self.config.target_type, self.config.num_labels
        y_true = preprocess_target(target_type, targets, num_labels=num_labels)
        return self.loss_func(logits, y_true)

    def compute_metric_inputs(self, batch: Any, output: OutputType):
        targets, logits = batch["labels"], output["logits"]
        target_type, num_labels = self.config.target_type, self.config.num_labels
        y_true = preprocess_target(target_type, targets, num_labels=num_labels)
        y_pred, y_score = preprocess_input(target_type, logits)
        pooler_output = output.get("pooler_output")
        return {
            "y_true": y_true,
            "y_pred": y_pred,
            "y_score": y_score,
            "embedding": pooler_output,
        }

    def predict_step(self, batch: Any, **kwargs) -> OutputType:
        kwargs = {}
        for k, v in batch.items():
            if k not in TRANSFORMERS_INPUTS:
                continue
            kwargs[k] = v
        return self(**kwargs)

    def set_num_labels(self, num_labels: int):
        original_device = self.device
        output_layer = torch.nn.Linear(self.config.hidden_size, num_labels)
        # PYTORCH_ENABLE_MPS_FALLBACK=1
        self.to("cpu")
        # copy old weights
        with torch.no_grad():
            output_layer.weight.index_copy_(
                0,
                torch.tensor(
                    range(self.classifier.output.weight.size(0)),
                    dtype=torch.long,
                ),
                self.classifier.output.weight,
            )
            output_layer.bias.index_copy_(
                0,
                torch.tensor(
                    range(self.classifier.output.bias.size(0)),
                    dtype=torch.long,
                ),
                self.classifier.output.bias,
            )
        # replace classifier
        self.classifier.output = output_layer
        self.config.num_labels = num_labels
        self.loss_func = TargetBasedLoss(self.config.target_type, num_labels=num_labels)
        # move back to original device
        self.to(original_device)
