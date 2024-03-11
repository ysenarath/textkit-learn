from __future__ import annotations

import functools
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.utils.data import DataLoader
from typing_extensions import ParamSpec, Self

from tklearn.base.model import ModelBase
from tklearn.metrics import Evaluator, Metric
from tklearn.nn.callbacks import ModelCallback, ModelCallbackList
from tklearn.nn.utils.data import RecordBatch, create_dataset
from tklearn.utils.array import concat, detach, move_to_device

P = ParamSpec("P")

TorchModule = TypeVar("TorchModule", bound=torch.nn.Module)

X, Y, Z = TypeVar("X"), TypeVar("Y"), TypeVar("Z")

XY = Union[
    X,
    Tuple[X, Y],
    None,
]


def get_available_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps" if not torch.backends.mps.is_built() else "cpu"
    return "cpu"


class Model(torch.nn.Module, ModelBase[X, Y, Z]):
    def to(self, device: Union[str, torch.device] = None, **kwargs) -> Model:
        if device is None:
            device = get_available_device()
        return super().to(device, **kwargs)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def fit(  # noqa: PLR0914
        self,
        x: XY,
        y: Y = None,
        /,
        batch_size: int = 32,
        epochs: int = 1,
        shuffle: bool = True,
        callbacks: Optional[Sequence[ModelCallback]] = None,
        validation_data: XY = None,
        metrics: Union[
            Evaluator, Sequence[Metric], Dict[str, Metric], None
        ] = None,
    ) -> Self:
        if not isinstance(callbacks, ModelCallbackList):
            callbacks = ModelCallbackList(callbacks)
        train_dataset = create_dataset(x, y)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=train_dataset.collate,
        )
        evaluate = functools.partial(
            self.evaluate,
            validation_data,
            metrics=metrics,
            prefix="valid_",
            callbacks=callbacks,
            batch_size=batch_size,
            return_dict=True,
        )
        device = self.device
        self.train()
        optimizer = self.configure_optimizers()
        lr_scheduler = self.configure_lr_scheduler(optimizer)
        params = {
            "batch_size": batch_size,
            "epochs": epochs,
            "steps": len(train_dataloader),
        }
        callbacks.set_params(params)
        callbacks.on_train_begin()
        epoch_logs = {}
        for epoch_idx in range(epochs):
            callbacks.on_epoch_begin(epoch_idx)
            batch_idx, total_loss = 0, 0.0
            for batch_idx, batch in enumerate(train_dataloader):
                callbacks.on_train_batch_begin(batch_idx)
                if not self.training:
                    self.train()
                batch = move_to_device(batch, device)
                batch_output = self.predict_on_batch(batch)
                loss = self.compute_loss(batch, batch_output)
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
            eval_results = evaluate()
            epoch_logs.update(eval_results)
            callbacks.on_epoch_end(epoch_idx, logs=epoch_logs)
        callbacks.on_train_end(epoch_logs)
        return self

    def predict(
        self,
        x: XY,
        y: Y = None,
        /,
        batch_size: int = 32,
        callbacks: Optional[Sequence[ModelCallback]] = None,
    ) -> Z:
        output = []
        for _, _, batch_output, _ in self.predict_on_batch_iter(
            x,
            y,
            batch_size=batch_size,
            callbacks=callbacks,
        ):
            output.append(batch_output)
        return concat(output)

    def extract_eval_input(  # noqa: PLR6301
        self, batch: RecordBatch, output: Z
    ) -> Dict[str, Any]:
        return {"y_true": batch.y, "y_pred": output}

    def evaluate(
        self,
        x: XY,
        y: Y = None,
        /,
        metrics: Union[
            Evaluator, Sequence[Metric], Dict[str, Metric], None
        ] = None,
        batch_size: int = 32,
        callbacks: Optional[Sequence[ModelCallback]] = None,
        include_loss: bool = True,
        return_dict: bool = False,
        prefix: str = "",
    ) -> Union[Dict[str, Any], Tuple[Any, ...]]:
        dataset = create_dataset(x, y)
        if dataset is None:
            return {} if return_dict else ()
        n_batches = 0
        total_loss = 0.0
        if not isinstance(metrics, Evaluator):
            metrics = Evaluator(metrics)
        metrics.reset()
        for _, batch, output, batch_loss in self.predict_on_batch_iter(
            dataset,
            callbacks=callbacks,
            batch_size=batch_size,
        ):
            n_batches += 1
            total_loss += batch_loss
            metrics.update_state(**self.extract_eval_input(batch, output))
        average_loss = total_loss / n_batches if n_batches > 0 else None
        if return_dict:
            return {
                **({f"{prefix}loss": average_loss} if include_loss else {}),
                **{
                    f"{prefix}{name}": value
                    for name, value in metrics.result(return_dict=True).items()
                },
            }
        return (*((average_loss,) if include_loss else ()), *metrics.result())

    def predict_on_batch_iter(
        self,
        x: XY,
        y: Y = None,
        /,
        callbacks: Optional[Sequence[ModelCallback]] = None,
        batch_size: int = 32,
    ) -> Generator[Tuple[int, RecordBatch, Z, float], None, None]:
        device = next(self.parameters()).device
        if not isinstance(callbacks, ModelCallbackList):
            callbacks = ModelCallbackList(callbacks)
        test_dataset = create_dataset(x, y)
        test_dataloader = DataLoader(
            test_dataset,
            shuffle=False,
            batch_size=batch_size,
            collate_fn=test_dataset.collate,
        )
        self.eval()
        callbacks.on_predict_begin()
        batch_idx, total_loss = 0, 0.0
        for batch_idx, batch in enumerate(test_dataloader):
            callbacks.on_predict_batch_begin(batch_idx)
            if self.training:
                self.eval()
            batch = move_to_device(batch, device)
            batch_output = self.predict_on_batch(batch)
            loss = self.compute_loss(batch, batch_output)
            batch_output = move_to_device(detach(batch_output), "cpu")
            batch_loss = loss.item()
            total_loss += batch_loss
            batch_logs = {}
            callbacks.on_predict_batch_end(batch_idx, logs=batch_logs)
            yield batch_idx, batch, batch_output, batch_loss
        n_batches = batch_idx + 1
        average_loss = total_loss / n_batches if n_batches > 0 else None
        logs = {
            "loss": average_loss,
        }
        callbacks.on_predict_end(logs)

    def predict_on_batch(self, batch: RecordBatch) -> Z:
        raise NotImplementedError

    def compute_loss(
        self,
        batch: RecordBatch,
        output: Z,
    ) -> torch.Tensor:
        raise NotImplementedError

    def configure_optimizers(self) -> Optimizer:
        raise NotImplementedError

    def configure_lr_scheduler(  # noqa: PLR6301
        self,
        optimizer: Optimizer,
    ) -> Union[LRScheduler, Any, None]:
        return None
