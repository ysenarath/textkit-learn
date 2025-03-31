from __future__ import annotations

from dataclasses import dataclass, is_dataclass
from typing import Any, Dict, Iterable, Optional, Tuple, get_type_hints

import torch
from torch.utils.data import DataLoader
from transformers.modeling_outputs import ModelOutput as BaseModelOutput
from typing_extensions import Literal

from tklearn.nn.base.module import Module
from tklearn.nn.callbacks import Callback
from tklearn.utils.array import move_to_device


@dataclass
class ModelOutput(BaseModelOutput):
    @classmethod
    def from_dict(cls, *args, **kwargs):
        if len(args) == 1 and is_dataclass(args[0]):
            data = {k: v for k, v in args[0].items()}
            data.update(kwargs)
        else:
            data = dict(*args, **kwargs)
        hints = get_type_hints(cls)
        data = {k: v for k, v in data.items() if k in hints}
        return cls(**data)


@dataclass
class SequenceClassifierOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


@dataclass
class SequenceClassifierOutputWithPooling(SequenceClassifierOutput):
    # loss, logits, hidden_states, attentions
    pooler_output: Optional[torch.FloatTensor] = None


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
        if similarity != "cosine":
            raise ValueError(
                f"similarity metric '{similarity}' is not supported"
            )
        self.similarity = CosineSimilarity()
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(
            weight=weight,
            size_average=size_average,
            ignore_index=ignore_index,
            reduce=reduce,
            reduction=reduction,
            label_smoothing=label_smoothing,
        )

    def forward(
        self, inputs: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        prototypes_map = get_prototype_map(inputs, targets)
        target_map = {
            label_id: proto_label_id
            for proto_label_id, label_id in enumerate(
                sorted(prototypes_map.keys())
            )
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

    def forward(
        self, inputs: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        prototypes_map = get_prototype_map(inputs, targets)
        target_map = {
            label_id: i
            for i, label_id in enumerate(sorted(prototypes_map.keys()))
        }
        mapped_targets = torch.tensor(
            [target_map[label_id.item()] for label_id in targets],
            device=inputs.device,
            dtype=torch.long,
        )
        prototypes = torch.stack(list(prototypes_map.values()))
        similarity = self.similarity(inputs, prototypes)
        return self.cross_entropy_loss(similarity, mapped_targets)


class CosineSimilarity(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.metric = torch.nn.CosineSimilarity(dim=-1)

    def forward(
        self, input1: torch.Tensor, input2: torch.Tensor
    ) -> torch.Tensor:
        # input1: N x D (batch size, D)
        # input2: M x D (number of prototypes, D)
        # output: N x M (batch size, number of prototypes)
        #   we have a similarity score for each pair of input1 and input2
        input1 = input1.unsqueeze(1).expand(-1, input2.size(0), -1)
        input2 = input2.expand(input1.size(0), -1, -1)
        return self.metric(input1, input2)


def get_prototype_map(
    embeddings: torch.Tensor, label_ids: torch.Tensor
) -> Dict[int, torch.Tensor]:
    prototypes = {}
    label_id: torch.Tensor
    for label_id in label_ids.unique():
        mask = torch.eq(label_ids, label_id)  # label_ids == label_id
        prototype = embeddings[mask].mean(dim=0)
        prototypes[label_id.item()] = prototype
    return prototypes


def compute_prototypes(
    model: Module,
    dataloader: DataLoader | Iterable,
    *,
    device: torch.device = None,
    return_updated_indices: bool = False,
) -> Tuple[torch.Tensor, set[int]] | torch.Tensor:
    if device is None:
        device = model.device
    dataloader_idx = 0
    pooler_outputs, targets = [], []
    for batch_idx, batch in enumerate(dataloader):
        batch = move_to_device(batch, device)
        output = model.predict_step(
            batch, batch_idx=batch_idx, dataloader_idx=dataloader_idx
        )
        pooler_output: torch.Tensor = output["pooler_output"]
        pooler_outputs.append(pooler_output)
        targets.append(batch["labels"])
    pooler_outputs = torch.concat(pooler_outputs).detach()
    targets = torch.concat(targets).detach()
    prototype_map = get_prototype_map(pooler_outputs, targets)
    max_label_id = max(prototype_map.keys())
    # infer hidden_size from the prototype_map.values()
    hidden_size = next(iter(prototype_map.values())).size(0)
    # max_label_id is zero-based so we need to add 1 as
    # the argument to torch.zeros is the size/length of the tensor
    prototypes = torch.zeros(max_label_id + 1, hidden_size)
    updated_indices = set()
    for label_id, prototype in prototype_map.items():
        prototypes[label_id] = prototype
        updated_indices.add(label_id)
    if return_updated_indices:
        return prototypes.to(device), updated_indices
    return prototypes.to(device)


def update_prototypes_(
    model: Module,
    dataloader: DataLoader | Iterable,
    device: torch.device | str = None,
):
    if not hasattr(model, "prototypes"):
        # not a prototype model so let's not do anything
        return
    prototypes = getattr(model, "prototypes", None)
    # prototypes are already computed so let's skip
    new_prototypes, updated_indices = compute_prototypes(
        model,
        dataloader,
        device=device,
        return_updated_indices=True,
    )
    # copy prior prototypes to the model
    if prototypes is not None:
        for i in range(len(prototypes)):
            if i in updated_indices:
                continue
            new_prototypes[i] = prototypes[i]
    setattr(model, "prototypes", new_prototypes)


class PrototypeCallback(Callback):
    def __init__(
        self,
        dataloader: DataLoader | Iterable,
        device: torch.device | str = None,
        prototypes: torch.Tensor | None = None,
    ) -> None:
        # if the model does not have an attribute "prototypes"
        # then we will not do anything
        super().__init__()
        self.dataloader = dataloader
        self.device = device
        self.prototypes = prototypes

    def _on_test_or_predict_begin(self):
        if not hasattr(self.model, "prototypes"):
            return
        # reset prototypes to the original prototypes
        self.model.prototypes = self.prototypes
        # update prototypes based on the current model
        # if there is no data supporting a prototype then this will pick that
        # from the original prototypes
        update_prototypes_(self.model, self.dataloader, device=self.device)

    def on_test_begin(self, logs=None):
        self._on_test_or_predict_begin()

    def on_predict_begin(self, logs=None):
        self._on_test_or_predict_begin()
