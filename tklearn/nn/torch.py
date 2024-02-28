from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    List,
    Optional,
    ParamSpec,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

from tklearn.nn.callbacks import TorchModelCallback, TorchModelCallbackList
from tklearn.nn.utils import TorchDataset
from tklearn.utils.array import concat, detach, move_to_device
from tklearn.utils.func import method

P = ParamSpec("P")
T = TypeVar("T")
TorchModule = TypeVar("TorchModule", bound=torch.nn.Module)


if TYPE_CHECKING:
    from tklearn.nn.torch import TorchTrainer

InputDataType = Union[Tuple[List[Any], List[Any]], List[Any]]


def get_available_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps" if not torch.backends.mps.is_built() else "cpu"
    return "cpu"


class Model(torch.nn.Module):
    def __init__(self, model: TorchModule):
        super(Model, self).__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def fit(
        self,
        x: Any,
        y: Any = None,
        /,
        batch_size: int = 32,
        epochs: int = 1,
        shuffle: bool = True,
        callbacks: Optional[Sequence[TorchModelCallback]] = None,
        **kwargs,
    ) -> TorchTrainer:
        callbacks = TorchModelCallbackList(callbacks)
        train_dataset = TorchDataset(x=x, y=y)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self.collate_fn,
        )
        device = next(self.parameters()).device
        model = move_to_device(self.model, device)
        model.train()
        optimizer = self.configure_optimizers()
        lr_scheduler = self.configure_lr_scheduler(optimizer)
        callbacks.on_train_begin()
        epoch_logs = {}
        for epoch_idx in range(epochs):
            callbacks.on_epoch_begin(epoch_idx)
            batch_idx, total_loss = 0, 0.0
            for batch_idx, batch in enumerate(train_dataloader):
                callbacks.on_train_batch_begin(batch_idx)
                if not model.training:
                    model.train()
                batch = move_to_device(batch, device)
                _, loss = self.predict_on_batch(batch)
                loss.backward()
                optimizer.step()
                if lr_scheduler is not None:
                    lr_scheduler.step()
                optimizer.zero_grad()
                total_loss += loss.item()
                batch_logs = {}
                callbacks.on_train_batch_end(batch_idx, logs=batch_logs)
            n_batches = batch_idx + 1
            epoch_logs = {
                "loss": (total_loss / n_batches) if n_batches > 0 else None,
            }
            callbacks.on_epoch_end(epoch_idx, logs=epoch_logs)
        callbacks.on_train_end(epoch_logs)

    def predict_proba(
        self,
        x: Any,
        y: Any = None,
        /,
        batch_size: int = 32,
        callbacks: Optional[Sequence[TorchModelCallback]] = None,
        **kwargs,
    ) -> Any:
        device = next(self.parameters()).device
        callbacks = TorchModelCallbackList(callbacks)
        test_dataset = TorchDataset(x=x, y=y)
        test_dataloader = DataLoader(
            test_dataset,
            shuffle=False,
            batch_size=batch_size,
            collate_fn=self.collate_fn,
        )
        self.model.eval()
        callbacks.on_predict_begin()
        batch_idx, total_loss = 0, 0.0
        output = None
        for batch_idx, batch in enumerate(test_dataloader):
            callbacks.on_predict_batch_begin(batch_idx)
            if self.model.training:
                self.model.eval()
            batch = move_to_device(batch, device)
            batch_output, loss = self.predict_on_batch(batch)
            batch_output = move_to_device(detach(batch_output), "cpu")
            output = concat([output, batch_output], axis=0)
            total_loss += loss.item()
            batch_logs = {}
            callbacks.on_predict_batch_end(batch_idx, logs=batch_logs)
        n_batches = batch_idx + 1
        average_loss = total_loss / n_batches if n_batches > 0 else None
        logs = {
            "loss": average_loss,
        }
        callbacks.on_predict_end(logs)
        return (output, average_loss)

    @method
    def predict_on_batch(
        self, x: Any, y: Any = None, /, **kwargs
    ) -> Tuple[Any, torch.Tensor]:
        raise NotImplementedError

    @method
    def collate_fn(self, batch: List[Any]) -> Any:
        return default_collate(batch)

    @method
    def configure_lr_scheduler(  # noqa: PLR6301
        self, optimizer: Optimizer
    ) -> Union[LRScheduler, Any, None]:
        return None

    @method
    def configure_optimizers(self) -> Optimizer:
        raise NotImplementedError
