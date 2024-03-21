from __future__ import annotations

import functools
import gc
from typing import (
    Any,
    Dict,
    Generator,
    Optional,
    Sequence,
    Tuple,
    TypedDict,
    TypeVar,
    Union,
)

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.utils.data import DataLoader
from typing_extensions import ParamSpec, Self, Unpack

from tklearn.base.model import ModelBase
from tklearn.metrics import Evaluator, Metric, create_evaluator
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


def empty_cache():
    torch.cuda.empty_cache()
    gc.collect()


def get_available_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps" if not torch.backends.mps.is_built() else "cpu"
    return "cpu"


class TrainingArgs(TypedDict):
    batch_size: int
    epochs: int
    shuffle: bool
    validation_data: XY
    validation_batch_size: Optional[int]
    metrics: Union[Evaluator, Sequence[Metric], Dict[str, Metric], None]
    optimizer: Union[str, Dict[str, Any], Optimizer]
    optimizer_args: Optional[Dict[str, Any]]
    lr_scheduler: Union[str, Dict[str, Any], LRScheduler, None]
    lr_scheduler_args: Optional[Dict[str, Any]]
    lr_scheduler_step: Optional[Dict[str, Any]]
    clip_grad_norm: Union[bool, float, Dict[str, Any], None]
    clip_grad_norm_args: Optional[Dict[str, Any]]
    callbacks: Optional[ModelCallbackList]


class Model(torch.nn.Module, ModelBase[X, Y, Z]):
    def to(
        self, device: Union[str, torch.device, None] = None, **kwargs: Any
    ) -> Model:
        if device is None:
            device = get_available_device()
        return super().to(device, **kwargs)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def stop_training(self) -> bool:
        if hasattr(self, "_stop_training"):
            return self._stop_training
        return False

    @stop_training.setter
    def stop_training(self, value: bool) -> None:
        self._stop_training = value

    def fit(
        self,
        x: XY,
        y: Y = None,
        /,
        **kwargs: Unpack[TrainingArgs],
    ) -> Self:
        self.training_args.clear()
        self.training_args.update(kwargs)
        batch_size = self.training_args.get("batch_size", 32)
        train_dataset = create_dataset(x, y)
        shuffle = self.training_args.get("shuffle", True)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=train_dataset.collate,
        )
        callbacks = self.training_args.get("callbacks", None)
        if not isinstance(callbacks, ModelCallbackList):
            callbacks = ModelCallbackList(callbacks)
        validation_data = self.training_args.get("validation_data")
        validation_batch_size = self.training_args.get(
            "validation_batch_size", batch_size
        )
        metrics = self.training_args.get("metrics")
        evaluate = functools.partial(
            self.evaluate,
            validation_data,
            metrics=metrics,
            prefix="valid_",
            callbacks=callbacks,
            batch_size=validation_batch_size,
            return_dict=True,
        )
        self.stop_training = False
        device = self.device
        self.train()
        optimizer = self._fit_configure_optimizers()
        steps = len(train_dataloader)
        lr_scheduler = self._fit_configure_lr_scheduler(optimizer)
        epochs = self.training_args.get("epochs", 1)
        params = {
            "batch_size": batch_size,
            "epochs": epochs,
            "steps": steps,
        }
        callbacks.set_params(params)
        callbacks.set_model(self)
        callbacks.on_train_begin()
        epoch_logs = {}
        for epoch_idx in range(epochs):
            callbacks.on_epoch_begin(epoch_idx)
            batch_idx, total_loss = 0, 0.0
            for batch_idx, batch in enumerate(train_dataloader):
                total_loss += self._fit_on_batch(
                    batch=batch,
                    batch_idx=batch_idx,
                    callbacks=callbacks,
                    device=device,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                )
                empty_cache()
            self._fit_lr_scheduler_step(lr_scheduler, "epoch")
            n_batches = batch_idx + 1
            epoch_logs = {
                "loss": (total_loss / n_batches) if n_batches > 0 else None,
            }
            eval_results = evaluate()
            epoch_logs.update(eval_results)
            callbacks.on_epoch_end(epoch_idx, logs=epoch_logs)
            if self.stop_training:
                break
        callbacks.on_train_end(epoch_logs)
        self.training_args.clear()
        return self

    def _fit_on_batch(
        self,
        batch: RecordBatch,
        *,
        device: Union[str, torch.device],
        optimizer: Optimizer,
        lr_scheduler: Union[LRScheduler, Any, None],
        callbacks: Optional[ModelCallbackList] = None,
        batch_idx: Optional[int] = None,
    ) -> float:
        if callbacks:
            callbacks.on_train_batch_begin(batch_idx)
        if not self.training:
            self.train()
        batch = move_to_device(batch, device)
        batch_output = self.predict_on_batch(batch)
        loss = self.compute_loss(batch, batch_output)
        optimizer.zero_grad()
        loss.backward()
        self._fit_clip_grad_norm()
        optimizer.step()
        self._fit_lr_scheduler_step(lr_scheduler, "batch")
        loss_value = loss.item()
        batch_logs = {}
        if callbacks:
            callbacks.on_train_batch_end(
                batch_idx,
                logs=batch_logs,
            )
        return loss_value

    def _fit_clip_grad_norm(self) -> None:
        clip_grad_norm = self.training_args.get("clip_grad_norm")
        if clip_grad_norm is False or clip_grad_norm is None:
            return
        clip_grad_norm_args: dict = self.training_args.get(
            "clip_grad_norm_args", {}
        )
        if isinstance(clip_grad_norm, (int, float)):
            clip_grad_norm_args["max_norm"] = clip_grad_norm
        elif isinstance(clip_grad_norm, dict):
            clip_grad_norm_args.update(clip_grad_norm)
        else:
            msg = (
                "clip_grad_norm should be a boolean, int, "
                f"float, or dict, but got {clip_grad_norm}"
            )
            raise ValueError(msg)
        torch.nn.utils.clip_grad_norm_(
            self.parameters(), **clip_grad_norm_args
        )

    def _fit_configure_optimizers(self) -> Optimizer:
        optimizer = self.training_args.get("optimizer")
        if isinstance(optimizer, str):
            optimizer_type = getattr(torch.optim, optimizer)
            optimizer_args = self.training_args.get("optimizer_args", {})
            return optimizer_type(
                self.parameters(),
                **optimizer_args,
            )
        if isinstance(optimizer, dict):
            optimizer_type = getattr(torch.optim, optimizer["$type"])
            optimizer_args = {
                k: v for k, v in optimizer.items() if not k.startswith("$")
            }
            return optimizer_type(
                self.parameters(),
                **optimizer_args,
            )
        return optimizer

    def _fit_configure_lr_scheduler(
        self, optimizer: Optimizer
    ) -> Union[LRScheduler, Any, None]:
        lr_scheduler = self.training_args.get("lr_scheduler")
        if isinstance(lr_scheduler, str):
            lr_scheduler_type = getattr(torch.optim.lr_scheduler, lr_scheduler)
            lr_scheduler_args = self.training_args.get("lr_schedule_args", {})
            return lr_scheduler_type(
                optimizer,
                **lr_scheduler_args,
            )
        if isinstance(lr_scheduler, dict):
            lr_scheduler_type = getattr(
                torch.optim.lr_scheduler, lr_scheduler["$type"]
            )
            lr_scheduler_args = {
                k: v for k, v in lr_scheduler.items() if not k.startswith("$")
            }
            return lr_scheduler_type(
                optimizer,
                **lr_scheduler_args,
            )
        return lr_scheduler

    def _fit_lr_scheduler_step(
        self,
        scheduler: Union[LRScheduler, Any, None],
        iter_type: str,
    ) -> None:
        if scheduler is None:
            return
        lr_scheduler_step: Optional[dict] = self.training_args.get(
            "lr_scheduler_step",
        )
        if lr_scheduler_step is None:
            scheduler.step()
            return
        if lr_scheduler_step.pop("on", "epoch") != iter_type:
            return
        scheduler.step(**lr_scheduler_step)

    def predict(
        self,
        x: XY,
        y: Y = None,
        /,
        batch_size: int = 32,
        callbacks: Optional[Sequence[ModelCallback]] = None,
    ) -> Z:
        output = []
        for _, _, batch_output, _ in self.predict_iter(
            x,
            y,
            batch_size=batch_size,
            callbacks=callbacks,
        ):
            output.append(batch_output)
        return concat(output)

    def predict_iter(
        self,
        x: XY,
        y: Y = None,
        /,
        batch_size: int = 32,
        callbacks: Optional[Sequence[ModelCallback]] = None,
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
            # clear memory cache
            del batch, batch_output, loss, batch_loss
            empty_cache()
        n_batches = batch_idx + 1
        average_loss = total_loss / n_batches if n_batches > 0 else None
        logs = {
            "loss": average_loss,
        }
        callbacks.on_predict_end(logs)

    def predict_on_batch(self, batch: RecordBatch) -> Z:
        raise NotImplementedError

    def evaluate(
        self,
        x: XY,
        y: Y = None,
        /,
        metrics: Union[
            Evaluator, Sequence[Metric], Dict[str, Metric], None
        ] = None,
        batch_size: int = 32,
        include_loss: bool = True,
        return_dict: bool = False,
        prefix: str = "",
        callbacks: Optional[Sequence[ModelCallback]] = None,
    ) -> Union[Dict[str, Any], Tuple[Any, ...]]:
        n_batches = 0
        total_loss = 0.0
        evaluator = create_evaluator(metrics)
        evaluator.reset()
        for _, batch, output, batch_loss in self.predict_iter(
            x,
            y,
            callbacks=callbacks,
            batch_size=batch_size,
        ):
            n_batches += 1
            total_loss += batch_loss
            eval_input = self.extract_eval_input(batch, output)
            evaluator.update_state(**eval_input)
        average_loss = (total_loss / n_batches) if n_batches > 0 else None
        if return_dict:
            return {
                **({f"{prefix}loss": average_loss} if include_loss else {}),
                **{
                    f"{prefix}{name}": value
                    for name, value in evaluator.result(
                        return_dict=True
                    ).items()
                },
            }
        return (
            *((average_loss,) if include_loss else ()),
            *evaluator.result(),
        )

    def extract_eval_input(  # noqa: PLR6301
        self, batch: RecordBatch, output: Z
    ) -> Dict[str, Any]:
        return {"y_true": batch.y, "y_pred": output}

    def compute_loss(self, batch: RecordBatch, output: Z) -> torch.Tensor:
        raise NotImplementedError

    @property
    def training_args(self) -> Dict[str, Any]:
        if not hasattr(self, "_training_args"):
            self._training_args = {}
        return self._training_args
