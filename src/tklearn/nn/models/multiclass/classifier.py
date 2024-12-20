from typing import ClassVar, Optional, overload, runtime_checkable

import torch
from torch import nn
from transformers.modeling_outputs import BaseModelOutput
from typing_extensions import Literal, Protocol

from tklearn.models.base import Model, ModelConfig
from tklearn.models.multiclass.helpers import (
    BatchPrototypeLoss,
    CosineSimilarity,
    SequenceClassifierOutputWithPooling,
)
from tklearn.nn.loss import TargetBasedLoss
from tklearn.nn.utils.preprocessing import preprocess_input, preprocess_target

__all__ = [
    "LinearLayerMulticlassClassifier",
    "LinearLayerMulticlassClassifierConfig",
    "PrototypeBasedMulticlassClassifier",
    "PrototypeBasedMulticlassClassifierConfig",
]


@runtime_checkable
class OutputWithPooling(Protocol):
    pooler_output: torch.Tensor

    @overload
    def __getitem__(self, key: Literal["pooler_output"]) -> torch.Tensor: ...


class LinearClassifierLayer(nn.Module):
    def __init__(
        self, hidden_size: int, num_labels: int = 0, dropout: float = 0.2
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()
        self.output = nn.Linear(hidden_size, num_labels)

    def forward(self, pooler_output, **kwargs):
        x = self.dropout(pooler_output)
        x = self.output(x)
        return x, pooler_output


class LinearLayerMulticlassClassifierConfig(ModelConfig):
    type: ClassVar[str] = "linear"
    output_dropout: float = 0.2
    num_labels: int = 0


class LinearLayerMulticlassClassifier(Model):
    config: LinearLayerMulticlassClassifierConfig

    def __post_init__(self) -> None:
        super().__post_init__()
        self.classifier = LinearClassifierLayer(
            hidden_size=self.hidden_size,
            num_labels=self.config.num_labels,
            dropout=self.config.output_dropout,
        )
        self.loss_func = TargetBasedLoss(
            "multiclass", num_labels=self.num_labels
        )

    @property
    def hidden_size(self) -> int:
        return self.backbone.hidden_size

    @property
    def num_labels(self) -> int:
        return self.classifier.output.out_features

    @num_labels.setter
    def num_labels(self, value: int):
        original_device = next(self.parameters()).device
        output_layer = torch.nn.Linear(self.hidden_size, value)
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
        # move back to original device
        self.to(original_device)
        # update num_labels of loss function
        self.loss_func = TargetBasedLoss(
            "multiclass", num_labels=self.num_labels
        )

    def predict_step(
        self, batch, **kwargs
    ) -> SequenceClassifierOutputWithPooling:
        outputs = self.backbone(batch)
        logits, pooler_output = self.classifier(outputs.pooler_output)
        # results.logits /= self.temperature
        output = SequenceClassifierOutputWithPooling.from_dict(
            outputs, logits=logits, pooler_output=pooler_output
        )
        return output

    def compute_loss(self, batch, output, **kwargs) -> torch.Tensor:
        targets, logits = batch["labels"], output["logits"]
        y_true = preprocess_target(
            "multiclass",
            targets,
            num_labels=self.num_labels,
        )
        return self.loss_func(logits, y_true)

    def compute_metric_inputs(self, batch, output, **kwargs) -> dict:
        targets, logits = batch["labels"], output["logits"]
        target_type, num_labels = (
            "multiclass",
            self.num_labels,
        )
        y_true = preprocess_target(target_type, targets, num_labels=num_labels)
        y_pred, y_score = preprocess_input(target_type, logits)
        pooler_output = output.get("pooler_output")
        return {
            "y_true": y_true,
            "y_pred": y_pred,
            "y_score": y_score,
            "embedding": pooler_output,
        }


class PrototypeBasedMulticlassClassifierConfig(ModelConfig):
    type: ClassVar[str] = "prototype"


class PrototypeBasedMulticlassClassifier(Model):
    config: PrototypeBasedMulticlassClassifierConfig

    def __post_init__(self) -> None:
        super().__post_init__()
        self.prototypes: Optional[torch.Tensor] = None
        self.loss_func = BatchPrototypeLoss()
        self.similarity = CosineSimilarity()

    def predict_step(self, batch, **kwargs) -> BaseModelOutput:
        return self.backbone(batch)

    def compute_loss(self, batch, output, **kwargs) -> torch.Tensor:
        targets, pooler_output = batch["labels"], output["pooler_output"]
        # dynamic number of labels therefore do not pass num_labels
        y_true = preprocess_target("multiclass", targets)
        targets = y_true.to(torch.long)
        return self.loss_func(pooler_output, targets)

    def compute_metric_inputs(self, batch, output, **kwargs) -> dict:
        targets, pooler_output = batch["labels"], output["pooler_output"]
        logits = self.similarity(pooler_output, self.prototypes)
        target_type = "multiclass"
        y_true = preprocess_target(target_type, targets)
        y_pred, y_score = preprocess_input(target_type, logits)
        return {
            "y_true": y_true,
            "y_pred": y_pred,
            "y_score": y_score,
            "embedding": pooler_output,
        }
