from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from functools import partial
from typing import Generic, TypeVar, overload

import numpy as np
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from typing_extensions import Literal

from tklearn.nn.base.module import Module
from tklearn.nn.base.predictor import iter_batch_outputs
from tklearn.nn.callbacks.base import Callback, CallbackList, CallbacksMixin
from tklearn.utils.array import move_to_device

__all__ = [
    "Encoder",
    "encode",
]

BatchT = TypeVar("BatchT")
OutputT = TypeVar("OutputT")
DEFAULT_BATCH_SIZE = 1000


def _to_format(
    value: torch.Tensor | np.ndarray | list, return_tensors: str | None
) -> torch.Tensor | np.ndarray | list:
    if isinstance(value, torch.Tensor):
        value = move_to_device(value.detach(), "cpu")
        if return_tensors == "np":
            return value.numpy()
        if return_tensors is None:
            return value.tolist()
        return value
    if isinstance(value, (np.ndarray, list)):
        if return_tensors == "pt":
            return torch.as_tensor(np.asarray(value))
        if return_tensors == "np":
            return np.asarray(value)
        return np.asarray(value).tolist()
    msg = (
        "expected the encoded output to be a tensor, array or list, "
        f"got {type(value).__name__}"
    )
    raise TypeError(msg)


class Encoder(CallbacksMixin, Generic[BatchT, OutputT]):
    """Encode batches into fixed-size vectors with a model.

    Parameters
    ----------
    model : Module
        The model; its `predict_step` output must contain `output_key`.
    output_key : str, default="pooler_output"
        The output field holding one vector per example.
    callbacks : Callback or iterable of Callback, optional
        Callbacks receiving the ``on_predict_*`` hooks.

    Examples
    --------
    >>> Encoder(model).encode(dataloader, return_tensors="np").shape
    (n_examples, hidden_size)
    """

    def __init__(
        self,
        model: Module[BatchT, OutputT],
        output_key: str = "pooler_output",
        callbacks: CallbackList | Iterable[Callback] | Callback | None = None,
    ) -> None:
        self.model = model
        self.output_key = output_key
        self.callbacks = callbacks

    @overload
    def encode(
        self,
        dataloader: Iterable[BatchT],
        return_tensors: Literal["pt"] = ...,
        return_list: Literal[False] = ...,
    ) -> torch.Tensor: ...
    @overload
    def encode(
        self,
        dataloader: Iterable[BatchT],
        return_tensors: Literal["np"],
        return_list: Literal[False] = ...,
    ) -> np.ndarray: ...
    @overload
    def encode(
        self,
        dataloader: Iterable[BatchT],
        return_tensors: Literal["pt"],
        return_list: Literal[True],
    ) -> list[torch.Tensor]: ...
    @overload
    def encode(
        self,
        dataloader: Iterable[BatchT],
        return_tensors: Literal["np"],
        return_list: Literal[True],
    ) -> list[np.ndarray]: ...
    @overload
    def encode(
        self,
        dataloader: Iterable[BatchT],
        return_tensors: None,
        return_list: Literal[True],
    ) -> list[list[float]]: ...
    def encode(
        self,
        dataloader: Iterable[BatchT],
        return_tensors: Literal["pt", "np"] | None = "pt",
        return_list: bool = False,
    ) -> torch.Tensor | np.ndarray | list:
        """Encode every example in a dataloader.

        Parameters
        ----------
        dataloader : iterable
            Batches to encode.
        return_tensors : {"pt", "np"} or None, default="pt"
            Type of each vector: torch tensor, numpy array, or (with None)
            a list of floats.
        return_list : bool, default=False
            Return a list with one vector per example instead of a single
            stacked tensor/array. Required when `return_tensors` is None.

        Returns
        -------
        torch.Tensor, np.ndarray or list
            The encodings, on the CPU.
        """
        if return_tensors not in {"pt", "np", None}:
            msg = (
                "return_tensors must be 'pt', 'np' or None, "
                f"got {return_tensors!r}"
            )
            raise ValueError(msg)
        if return_tensors is None and not return_list:
            msg = "return_tensors=None requires return_list=True"
            raise ValueError(msg)
        encodings = []
        for _, output in iter_batch_outputs(
            self.model, dataloader, self.callbacks, stage="predict"
        ):
            encodings.extend(
                _to_format(output[self.output_key], return_tensors)
            )
        if return_list:
            return encodings
        if return_tensors == "np":
            return np.asarray(encodings)
        return torch.stack(encodings)


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
    output_column_name: str,
    **kwargs,
) -> dict[str, list[torch.Tensor]]:
    batch: BatchDataset = BatchDataset(batch, length=len(indices))
    dataloader = DataLoader(
        batch,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin_memory,
        **kwargs,
    )
    encoder = Encoder(model)
    return {
        output_column_name: encoder.encode(
            dataloader, return_tensors="pt", return_list=True
        )
    }


def encode(
    dataset: Dataset,
    model: Module,
    batch_size: int = DEFAULT_BATCH_SIZE,
    pin_memory: bool = True,
    desc: str = "Encoding dataset",
    encode_batch_size: int = 32,
    collate_fn: Callable | None = None,
    output_column_name: str | None = None,
    **kwargs,
) -> Dataset:
    """
    Encodes a dataset using a given model.

    This function processes a dataset in chunks, encoding each chunk using the
    provided model. It supports batching, custom collation functions, and other
    configurations to optimize the encoding process.

    Parameters
    ----------
    dataset : Dataset
        The dataset to be encoded.
    model : Module
        The model used for encoding the dataset.
    batch_size : int, optional
        The size of the batches to process the dataset, by default DEFAULT_BATCH_SIZE.
    pin_memory : bool, optional
        If True, the data loader will copy tensors into CUDA pinned memory, by default True.
    desc : str, optional
        A description for the progress bar, by default "Encoding dataset".
    encode_batch_size : int, optional
        The batch size used for encoding within each chunk, by default 32.
    collate_fn : Callable or None, optional
        A function to merge a list of samples into a batch, by default None.
    output_column_name : str, optional
        The name of the column to store the encoded embeddings, by default "embedding".
    **kwargs
        Additional keyword arguments passed to the dataset's `map` function.

    Returns
    -------
    Dataset
        The encoded dataset.
    """
    if collate_fn is not None:
        kwargs["collate_fn"] = collate_fn
    if output_column_name is None:
        output_column_name = "embedding"
    model.eval()
    encoded_dataset = dataset.map(
        partial(
            _encode_chunk,
            model=model,
            batch_size=encode_batch_size,
            pin_memory=pin_memory,
            output_column_name=output_column_name,
            **kwargs,
        ),
        batched=True,
        batch_size=batch_size,
        desc=desc,
        with_indices=True,
    )
    return encoded_dataset
