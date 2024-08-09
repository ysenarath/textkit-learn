from __future__ import annotations

from typing import Any, Dict

import torch
from typing_extensions import Literal, Protocol, runtime_checkable


def get_prototypes(
    embeddings: torch.Tensor, label_ids: torch.Tensor
) -> Dict[int, torch.Tensor]:
    prototypes = {}
    label_id: torch.Tensor
    for label_id in label_ids.unique():
        mask = torch.eq(label_ids, label_id)  # label_ids == label_id
        prototype = embeddings[mask].mean(dim=0)
        prototypes[label_id.item()] = prototype
    return prototypes


@runtime_checkable
class OutputWithPooling(Protocol):
    pooler_output: torch.Tensor


class CosineSimilarity(torch.nn.Module):
    def __init__(self):
        self.metric = torch.nn.CosineSimilarity(dim=-1)

    def forward(self, input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
        # input1: N x D (batch size, D)
        # input2: M x D (number of prototypes, D)
        # output: N x M (batch size, number of prototypes)
        #   we have a similarity score for each pair of input1 and input2
        input1 = input1.unsqueeze(1).expand(-1, input2.size(0), -1)
        input2 = input2.expand(input1.size(0), -1, -1)
        return self.metric(input1, input2)


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
        prototypes_map = get_prototypes(inputs, targets)
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
        prototypes_map = get_prototypes(inputs, targets)
        target_map = {
            label_id: i for i, label_id in enumerate(sorted(prototypes_map.keys()))
        }
        mapped_targets = torch.tensor(
            [target_map[label_id.item()] for label_id in targets],
            device=inputs.device,
            dtype=torch.long,
        )
        prototypes_map = torch.stack(list(prototypes_map.values()))
        similarity = self.similarity(inputs, prototypes_map)
        return self.cross_entropy_loss(similarity, mapped_targets)
