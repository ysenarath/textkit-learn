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

from tklearn.base.trainer import Trainer
from tklearn.nn.callbacks import TorchTrainerCallback, TorchTrainerCallbackList
from tklearn.nn.utils import TorchDataset
from tklearn.utils.array import concat, detach, move_to_device, to_numpy
from tklearn.utils.func import method

P = ParamSpec("P")
T = TypeVar("T")
TorchModule = TypeVar("TorchModule", bound=torch.nn.Module)


if TYPE_CHECKING:
    from tklearn.nn.torch import TorchTrainer

InputDataType = Union[Tuple[List[Any], List[Any]], List[Any]]


class TorchEvaluator:
    def __init__(
        self,
        x: Any,
        y: Any = None,
        batch_size: int = 32,
        shuffle: bool = True,
    ):
        self.dataset = TorchDataset(x=x, y=y)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._y_pred = None

    def evaluate(self, trainer: TorchTrainer):
        if not isinstance(trainer, TorchTrainer):
            msg = (
                f"{type(trainer).__name__} is not an instance of TorchTrainer"
            )
            raise ValueError(msg)
        dataloader = DataLoader(
            self.dataset,
            shuffle=self.shuffle,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
        )
        for batch_idx, batch in enumerate(dataloader):
            y_pred, loss = trainer.predict(batch)
            # move y_pred to cpu and convert to numpy
            y_pred = to_numpy(move_to_device(y_pred, "cpu"))
            if self._y_pred is None:
                self._y_pred = y_pred
            else:
                self._y_pred = concat([self._y_pred, y_pred], axis=0)
        return {"loss": loss, "y_pred": self._y_pred}


def get_available_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps" if not torch.backends.mps.is_built() else "cpu"
    return "cpu"


class TorchTrainer(Trainer[TorchModule]):
    def __init__(
        self,
        model: TorchModule,
        num_epochs: int = 1,
        shuffle: bool = True,
        batch_size: int = 32,
        device: Optional[str] = None,
        callbacks: Optional[Sequence[TorchTrainerCallback]] = None,
    ) -> None:
        super().__init__(model=model)
        self.num_epochs = num_epochs
        self.shuffle = shuffle
        self.batch_size = batch_size
        if device is None:
            device = get_available_device()
        self.device: Union[str, torch.device] = device
        if callbacks is None:
            callbacks = []
        self.callbacks.extend(callbacks)

    @property
    def callbacks(self) -> TorchTrainerCallbackList:
        if not hasattr(self, "_callbacks"):
            self._callbacks = TorchTrainerCallbackList()
            self._callbacks.set_trainer(self)
        return self._callbacks

    def fit(
        self,
        x: Any,
        y: Any = None,
        /,
        evaluator: TorchEvaluator = None,
        callbacks: Optional[Sequence[TorchTrainerCallback]] = None,
        **kwargs,
    ) -> TorchTrainer:
        callbacks = self.callbacks.extend(
            callbacks,
            inplace=False,
        )
        train_dataset = TorchDataset(x=x, y=y)
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
        callbacks.on_train_begin()
        epoch_logs = {}
        for epoch_idx in range(self.num_epochs):
            callbacks.on_epoch_begin(epoch_idx)
            batch_idx, total_loss = 0, 0.0
            for batch_idx, batch in enumerate(train_dataloader):
                callbacks.on_train_batch_begin(batch_idx)
                if not self.model.training:
                    self.model.train()
                batch = move_to_device(batch, self.device)
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
            if evaluator is not None:
                eval_logs = evaluator.evaluate(self)
                epoch_logs.update(eval_logs)
            callbacks.on_epoch_end(epoch_idx, logs=epoch_logs)
        callbacks.on_train_end(epoch_logs)

    def predict_proba(
        self,
        x: Any,
        y: Any = None,
        batch_size: Optional[int] = None,
        shuffle: Optional[bool] = None,
        /,
        callbacks: Optional[Sequence[TorchTrainerCallback]] = None,
        **kwargs,
    ) -> Any:
        callbacks = self.callbacks.extend(
            callbacks,
            inplace=False,
        )
        if batch_size is None:
            batch_size = self.batch_size
        if shuffle is None:
            shuffle = self.shuffle
        test_dataset = TorchDataset(x=x, y=y)
        test_dataloader = DataLoader(
            test_dataset,
            shuffle=shuffle,
            batch_size=batch_size,
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
            batch = move_to_device(batch, self.device)
            batch_output, loss = self.predict_on_batch(batch)
            batch_output = move_to_device(detach(batch_output), "cpu")
            output = concat([output, batch_output], axis=0)
            total_loss += loss.item()
            batch_logs = {}
            self.callbacks.on_predict_batch_end(batch_idx, logs=batch_logs)
        n_batches = batch_idx + 1
        average_loss = total_loss / n_batches if n_batches > 0 else None
        logs = {
            "loss": average_loss,
        }
        self.callbacks.on_predict_end(logs)
        return (output, average_loss)

    @method
    def predict_on_batch(
        self,
        x: Any,
        y: Any = None,
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
