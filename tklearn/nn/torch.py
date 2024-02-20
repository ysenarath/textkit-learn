from __future__ import annotations

import tempfile
from typing import (
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

from tklearn.core.trainer import Trainer
from tklearn.nn.callbacks.callback import TrainerCallback, TrainerCallbackList
from tklearn.nn.dataset import TorchDataset
from tklearn.utils.array import concat, move_to_device
from tklearn.utils.func import method

P = ParamSpec("P")
TorchModule = torch.nn.Module
T = TypeVar("T")


def copy_model(model: TorchModule) -> None:
    with tempfile.TemporaryFile(suffix=".pth") as f:
        torch.save(model, f)
        return torch.load(f)


class TorchTrainer(Trainer[TorchModule]):
    def __init__(
        self,
        model: TorchModule,
        num_epochs: int,
        shuffle: bool = True,
        callbacks: Optional[Sequence[TrainerCallback]] = None,
        batch_size: int = 8,
        device: str = "cuda",
    ) -> None:
        super().__init__(model)
        self.num_epochs = num_epochs
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.device = device
        callbacks = TrainerCallbackList(callbacks)
        callbacks.set_trainer(self)
        self._callbacks = callbacks

    @property
    def callbacks(self) -> TrainerCallbackList:
        return self._callbacks

    def fit(self, data: Any, target: Any = None, **kwargs) -> TorchTrainer:
        # this will create a TorchDataset, if data is not a TorchDataset
        # and target is not None
        train_dataset = TorchDataset(data=data, target=target, **kwargs)
        train_dataloader = DataLoader(
            train_dataset,
            shuffle=self.shuffle,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
        )
        self.model = move_to_device(self.model, self.device)
        self.model.train()
        optimizer = self.configure_optimizers()
        lr_scheduler = self.configure_lr_scheduler(optimizer)
        self.callbacks.on_train_begin()
        epoch_logs = {}
        for epoch_idx in range(self.num_epochs):
            self.callbacks.on_epoch_begin(epoch_idx)
            batch_idx, total_loss = 0, 0.0
            for batch_idx, batch in enumerate(train_dataloader):
                self.callbacks.on_train_batch_begin(batch_idx)
                if not self.model.training:
                    self.model.train()
                _, loss = self.training_step(
                    batch,
                    batch_idx=batch_idx,
                    epoch_idx=epoch_idx,
                )
                loss.backward()
                optimizer.step()
                if lr_scheduler is not None:
                    lr_scheduler.step()
                optimizer.zero_grad()
                total_loss += loss.item()
                batch_logs = {}
                self.callbacks.on_train_batch_end(batch_idx, logs=batch_logs)
            n_batches = batch_idx + 1
            epoch_logs = {
                "loss": (total_loss / n_batches) if n_batches > 0 else None,
            }
            self.callbacks.on_epoch_end(epoch_idx, logs=epoch_logs)
        self.callbacks.on_train_end(epoch_logs)

    def _predict(self, data: Any, **kwargs) -> Any:
        test_dataset = TorchDataset(data=data, **kwargs)
        test_dataloader = DataLoader(
            test_dataset,
            shuffle=self.shuffle,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
        )
        self.model = move_to_device(self.model, self.device)
        self.model.eval()
        self.callbacks.on_predict_begin()
        batch_idx, total_loss = 0, 0.0
        output = None
        for batch_idx, batch in enumerate(test_dataloader):
            self.callbacks.on_predict_batch_begin(batch_idx)
            if self.model.training:
                self.model.eval()
            batch_output, loss = self.training_step(
                batch,
                batch_idx=batch_idx,
            )
            output = concat([output, batch_output], axis=0)
            total_loss += loss.item()
            batch_logs = {}
            self.callbacks.on_predict_batch_end(batch_idx, logs=batch_logs)
        n_batches = batch_idx + 1
        logs = {
            "loss": (total_loss / n_batches) if n_batches > 0 else None,
        }
        self.callbacks.on_predict_end(logs)
        return output

    # --------------------- SUPPORT FUNCTIONS ---------------------

    @method
    def collate_fn(self, batch: List[Any]) -> TorchDataset:
        batch = default_collate(batch)
        # batch = move_to_device(batch, self.device)
        return TorchDataset(**batch).to(self.device)

    @method
    def configure_lr_scheduler(  # noqa: PLR6301
        self, optimizer: Optimizer
    ) -> Union[LRScheduler, Any, None]:
        return None

    @method
    def configure_optimizers(self) -> Optimizer:
        raise NotImplementedError

    @method
    def training_step(
        self,
        batch: TorchDataset,
        /,
        batch_idx: int,
        epoch_idx: int = 0,
    ) -> Tuple[Any, torch.Tensor]:
        raise NotImplementedError
