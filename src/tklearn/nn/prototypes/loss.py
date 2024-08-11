from __future__ import annotations

from typing import Any

import torch
from typing_extensions import Literal, Protocol, runtime_checkable

from tklearn.nn.prototypes.helpers import CosineSimilarity, get_prototype_map


@runtime_checkable
class OutputWithPooling(Protocol):
    pooler_output: torch.Tensor


class BatchPrototypeLoss(torch.nn.Module):
    def __init__(
        self,
        similarity: Literal["cosine"] = "cosine",
        weight: torch.Tensor | None = None,
        size_average: Any | None = None,
        ignore_index: int = -100,
        reduce: Any | None = None,
        reduction: str = "mean",
        label_smoothing: float = 0,
    ):
        super().__init__()
        self.similarity = CosineSimilarity()
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(
            weight=weight,
            size_average=size_average,
            ignore_index=ignore_index,
            reduce=reduce,
            reduction=reduction,
            label_smoothing=label_smoothing,
        )

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        prototypes_map = get_prototype_map(inputs, targets)
        target_map = {
            label_id: proto_label_id
            for proto_label_id, label_id in enumerate(sorted(prototypes_map.keys()))
        }
        mapped_targets = torch.tensor(
            [target_map[label_id.item()] for label_id in targets],
            device=inputs.device,
            dtype=torch.long,
        )
        prototypes_map = torch.stack(list(prototypes_map.values()))
        similarity = self.similarity(inputs, prototypes_map)
        return self.cross_entropy_loss(similarity, mapped_targets)


class ClassPrototypeLoss(torch.nn.Module):
    def __init__(
        self,
        similarity: Literal["cosine"] = "cosine",
        weight: torch.Tensor | None = None,
        size_average: Any | None = None,
        ignore_index: int = -100,
        reduce: Any | None = None,
        reduction: str = "mean",
        label_smoothing: float = 0,
    ):
        super().__init__()
        self.prototypes = torch.nn.ParameterDict()
        self.similarity = CosineSimilarity()
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(
            weight=weight,
            size_average=size_average,
            ignore_index=ignore_index,
            reduce=reduce,
            reduction=reduction,
            label_smoothing=label_smoothing,
        )

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        prototypes_map = get_prototype_map(inputs, targets)
        target_map = {
            label_id: i for i, label_id in enumerate(sorted(prototypes_map.keys()))
        }
        mapped_targets = torch.tensor(
            [target_map[label_id.item()] for label_id in targets],
            device=inputs.device,
            dtype=torch.long,
        )
        prototypes = torch.stack(list(prototypes_map.values()))
        similarity = self.similarity(inputs, prototypes)
        return self.cross_entropy_loss(similarity, mapped_targets)
