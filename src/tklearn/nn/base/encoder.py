from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from functools import partial
from typing import Generator, Generic, TypeVar, Union, overload

import numpy as np
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from typing_extensions import Literal

from tklearn.nn.base.module import Module
from tklearn.nn.callbacks.base import Callback, CallbackList, CallbacksMixin
from tklearn.nn.loss import LossDict
from tklearn.utils.array import move_to_device

K = TypeVar("K")
V = TypeVar("V")
L = Union[torch.Tensor, Mapping[str, torch.Tensor], LossDict, None]
DEFAULT_BATCH_SIZE = 1000


class Encoder(CallbacksMixin, Generic[K, V]):
    def __init__(
        self,
        model: Module[K, V],
        dataloader: DataLoader,
        callbacks: CallbackList | Iterable[Callback] | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.dataloader = dataloader
        self.callbacks = callbacks

    @torch.no_grad()
    def iter_batches(
        self,
    ) -> Generator[tuple[int, K, V, LossDict], None, None]:
        if self.model.training:
            self.model.eval()

        # set the callback params
        callback_params = {}

        if self.callbacks.params is not None:
            callback_params.update(self.callbacks.params)

        callback_params.update({"pred_steps": len(self.dataloader)})

        self.callbacks.set_params(callback_params)
        self.callbacks.set_model(self.model)

        # start the prediction
        self.callbacks.on_predict_begin()

        dataloader_idx = None
        for batch_idx, batch in enumerate(self.dataloader):
            batch = move_to_device(batch, self.model.device, non_blocking=True)
            self.callbacks.on_predict_batch_begin(batch_idx)
            output = self.model.predict_step(
                batch, batch_idx=batch_idx, dataloader_idx=dataloader_idx
            )
            self.callbacks.on_predict_batch_end(batch_idx, logs={})
            yield batch_idx, batch, output, None

        self.callbacks.on_predict_end()

    @overload
    def encode(
        self,
        return_tensors: Literal["pt"] | None,
        return_list: Literal[False],
    ) -> torch.Tensor: ...
    @overload
    def encode(
        self,
        return_tensors: Literal["np"],
        return_list: Literal[False],
    ) -> np.ndarray: ...
    @overload
    def encode(
        self,
        return_tensors: Literal["pt"],
        return_list: Literal[True],
    ) -> list[torch.Tensor]: ...
    @overload
    def encode(
        self,
        return_tensors: Literal["np"],
        return_list: Literal[True],
    ) -> list[np.ndarray]: ...
    @overload
    def encode(
        self,
        return_tensors: None,
        return_list: Literal[True],
    ) -> list[list[float]]: ...
    def encode(
        self,
        return_tensors: str | None = "pt",
        return_list: bool = False,
    ) -> torch.Tensor | np.ndarray | list[torch.Tensor | np.ndarray]:
        if return_tensors is None:
            if not return_list:
                return_tensors = "pt"
        elif return_tensors not in {"pt", "np"}:
            ERR = f"return_tensors must be either 'pt' or 'np', got {return_tensors}."
            raise ValueError(ERR)

        self.model.eval()

        encodings = []

        for _, _, output, _ in self.iter_batches():
            pooler_output = output["pooler_output"]  # tensor in device
            if isinstance(pooler_output, torch.Tensor):
                pooler_output = pooler_output.detach()
                pooler_output = move_to_device(pooler_output, device="cpu")
                if return_tensors == "np":
                    pooler_output = pooler_output.numpy()
                elif return_tensors is None:
                    pooler_output = pooler_output.tolist()
            elif isinstance(pooler_output, np.ndarray):
                if return_tensors == "pt":
                    pooler_output = torch.from_numpy(pooler_output)
                elif return_tensors is None:
                    pooler_output = pooler_output.tolist()
            elif isinstance(pooler_output, list):
                if return_tensors == "pt":
                    pooler_output = torch.from_numpy(pooler_output)
                elif return_tensors == "np":
                    pooler_output = np.asarray(pooler_output)
            else:
                ERR = "expected '{k}' to be a `{e}`, got `{t}` instead".format(
                    k="pooler_output",
                    e=torch.Tensor.__name__,
                    t=type(pooler_output).__name__,
                )
                raise TypeError(ERR)
            encodings.extend(pooler_output)
            del pooler_output

        if return_list:
            pass  # do not convert to tensor or numpy array
        elif return_tensors == "np":
            encodings = np.asarray(encodings)
        elif return_tensors == "pt":
            encodings = torch.stack(encodings)

        return encodings


class BatchDataset(Sequence[Mapping[str, torch.Tensor]]):
    def __init__(self, data: Mapping[str, torch.Tensor], length: int) -> None:
        self.data = data
        self.length = length

    def __getitem__(self, index: int) -> Mapping[str, torch.Tensor]:
        return {k: v[index] for k, v in self.data.items()}

    def __len__(self) -> int:
        return self.length


def _encode_chunk(
    batch: dict[str, torch.Tensor],
    indices: list[int],
    model: Module,
    batch_size: int,
    pin_memory: bool,
) -> dict[str, list[torch.Tensor]]:
    batch: BatchDataset = BatchDataset(batch, length=len(indices))
    dataloader = DataLoader(
        batch,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin_memory,
    )
    encoder = Encoder(model, dataloader)
    return {
        "encodings": encoder.encode(
            return_tensors="pt",
            return_list=True,
        )
    }


def encode(
    dataset: Dataset,
    model: Module,
    batch_size: int = DEFAULT_BATCH_SIZE,
    pin_memory: bool = True,
    desc: str = "Encoding dataset",
    encode_batch_size: int = 32,
) -> Dataset:
    model.eval()
    encoded_dataset = dataset.map(
        partial(
            _encode_chunk,
            model=model,
            batch_size=encode_batch_size,
            pin_memory=pin_memory,
        ),
        batched=True,
        batch_size=batch_size,
        desc=desc,
        with_indices=True,
    )
    return encoded_dataset
