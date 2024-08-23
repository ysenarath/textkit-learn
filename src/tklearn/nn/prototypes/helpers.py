from __future__ import annotations

from typing import Dict, Iterable

import torch
from torch.utils.data import DataLoader

from tklearn.nn.base.module import Module
from tklearn.nn.callbacks import Callback
from tklearn.utils.array import move_to_device


class CosineSimilarity(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.metric = torch.nn.CosineSimilarity(dim=-1)

    def forward(self, input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
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
    model: Module, dataloader: DataLoader | Iterable, *, device: torch.device = None
) -> torch.Tensor:
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
    for label_id, prototype in prototype_map.items():
        prototypes[label_id] = prototype
    return prototypes.to(device)


class PrototypeCallback(Callback):
    def __init__(
        self,
        dataloader: DataLoader | Iterable,
        device: torch.device | str = None,
        prototypes: torch.Tensor = None,
    ) -> None:
        # if the model does not have an attribute "prototypes"
        # then we will not do anything
        super().__init__()
        self.dataloader = dataloader
        self.device = device
        self.prototypes = prototypes

    def on_predict_begin(self, logs=None):
        if not hasattr(self.model, "prototypes"):
            # not a prototype model so let's not do anything
            return
        # prototypes are already computed so let's skip
        prototypes = compute_prototypes(self.model, self.dataloader, device=self.device)
        # copy prior prototypes to the model
        if self.prototypes is not None:
            for i, prototype in enumerate(prototypes):
                prototypes[i] = prototype
        setattr(self.model, "prototypes", prototypes)
