from __future__ import annotations

import functools
import gc
from collections.abc import Mapping
from contextvars import ContextVar, copy_context
from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    Generator,
    Generic,
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
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import Sampler
from typing_extensions import ParamSpec, Self, Unpack
from werkzeug.local import LocalProxy

from tklearn.nn.callbacks import Callback, CallbackList
from tklearn.nn.loss import LossDict
from tklearn.nn.metrics import Evaluator, Metric
from tklearn.nn.utils.array import concat, move_to_device
from tklearn.nn.utils.data import Dataset, Record, RecordBatch

P = ParamSpec("P")

X, Y, Z = TypeVar("X"), TypeVar("Y"), TypeVar("Z")

XY = Union[X, Tuple[X, Y], None]

_training_ctx_var = ContextVar("training_ctx", default=None)
training_ctx: TrainingContext = LocalProxy(_training_ctx_var)


def empty_cache():
    torch.cuda.empty_cache()
    gc.collect()


def get_available_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    try:
        if torch.backends.mps.is_available():
            return "mps" if not torch.backends.mps.is_built() else "cpu"
    except AttributeError:
        pass
    return "cpu"


class TrainingArgsDict(TypedDict):
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
    callbacks: Optional[CallbackList]
    sampler: Optional[Sampler]
    batch_sampler: Optional[Sampler]


@dataclass
class TrainingContext:
    args: Mapping[str, Any]
    optimizer: Optimizer = None
    lr_scheduler: Optional[LRScheduler] = None


class Module(torch.nn.Module, Generic[X, Y, Z]):
    def to(
        self, device: Union[str, torch.device, None] = None, **kwargs: Any
    ) -> Module:
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

    def fit(self, x: XY, y: Y = None, /, **kwargs: Unpack[TrainingArgsDict]) -> Self:
        def run():
            ctx = TrainingContext(args=kwargs)
            token = _training_ctx_var.set(ctx)
            try:
                return self._fit_with_context(x, y)
            finally:
                _training_ctx_var.reset(token)

        return copy_context().run(run)

    def _fit_with_context(self, x: XY, y: Y = None) -> Self:
        training_args = training_ctx.args
        batch_size = training_args.get("batch_size", 32)
        train_dataset = Dataset(x, y)
        shuffle = training_args.get("shuffle", True)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self.collate_fn,
            pin_memory=True,
            sampler=training_args.get("sampler", None),
            batch_sampler=training_args.get("batch_sampler", None),
        )
        callbacks = training_args.get("callbacks", None)
        if not isinstance(callbacks, CallbackList):
            callbacks = CallbackList(callbacks)
        validation_data = training_args.get("validation_data")
        validation_batch_size = training_args.get("validation_batch_size", batch_size)
        metrics = training_args.get("metrics")
        evaluate = (
            functools.partial(
                self.evaluate,
                validation_data,
                metrics=metrics,
                prefix="valid_",
                callbacks=callbacks,
                batch_size=validation_batch_size,
                return_dict=True,
            )
            if validation_data is not None and metrics is not None
            else None
        )
        self.stop_training = False
        device = self.device
        self.train()
        self._fit_configure_optimizers()
        steps = len(train_dataloader)
        self._fit_configure_lr_scheduler()
        epochs = training_args.get("epochs", 1)
        params = {"batch_size": batch_size, "epochs": epochs, "steps": steps}
        callbacks.set_params(params)
        callbacks.set_model(self)
        callbacks.on_train_begin()
        epoch_logs = {}
        for epoch_idx in range(epochs):
            callbacks.on_epoch_begin(epoch_idx)
            batch_idx, total_loss_dict = 0, None
            for batch_idx, batch in enumerate(train_dataloader):
                batch_loss_dict = self._fit_on_batch(
                    batch=batch,
                    batch_idx=batch_idx,
                    callbacks=callbacks,
                    device=device,
                )
                total_loss_dict = batch_loss_dict + total_loss_dict
            self._fit_lr_scheduler_step("train")
            n_batches = batch_idx + 1
            if n_batches > 0 and total_loss_dict is not None:
                epoch_logs = total_loss_dict / n_batches
                epoch_logs = epoch_logs.item().to_dict()
            else:
                epoch_logs = {}
            if evaluate is not None:
                eval_results = evaluate()
                epoch_logs.update(eval_results)
            callbacks.on_epoch_end(epoch_idx, logs=epoch_logs)
            if self.stop_training:
                break
        callbacks.on_train_end(epoch_logs)
        return self

    def _fit_on_batch(
        self,
        batch: RecordBatch,
        *,
        device: Union[str, torch.device],
        callbacks: Optional[CallbackList] = None,
        batch_idx: Optional[int] = None,
    ) -> LossDict[torch.Tensor]:
        batch = move_to_device(batch, device, non_blocking=True)
        if callbacks:
            callbacks.on_train_batch_begin(batch_idx)
        if not self.training:
            self.train()
        batch_output = self.predict_on_batch(batch)
        loss = self.compute_loss(batch, batch_output)
        loss_dict = LossDict.from_loss(loss)
        training_ctx.optimizer.zero_grad()
        loss_dict.loss.backward()
        self._fit_clip_grad_norm()
        training_ctx.optimizer.step()
        self._fit_lr_scheduler_step("batch")
        batch_logs = {}
        if callbacks:
            callbacks.on_train_batch_end(
                batch_idx,
                logs=batch_logs,
            )
        return loss_dict.detach()

    def _fit_clip_grad_norm(self) -> None:
        clip_grad_norm = training_ctx.args.get("clip_grad_norm")
        if clip_grad_norm is False or clip_grad_norm is None:
            # do not clip gradients
            return
        clip_grad_norm_args = training_ctx.args.get("clip_grad_norm_args", {})
        if isinstance(clip_grad_norm, (int, float, bool)):
            clip_grad_norm_args["max_norm"] = float(clip_grad_norm)
        elif isinstance(clip_grad_norm, dict):
            clip_grad_norm_args.update(clip_grad_norm)
        else:
            msg = (
                "clip_grad_norm should be a boolean, int, float, or dict, but got "
                f"'{clip_grad_norm.__class__.__name__}'"
            )
            raise ValueError(msg)
        torch.nn.utils.clip_grad_norm_(
            self.parameters(),
            **clip_grad_norm_args,
        )

    def _fit_configure_optimizers(self):
        optimizer = training_ctx.args.get("optimizer")
        if isinstance(optimizer, str):
            optimizer_type = getattr(torch.optim, optimizer)
            optimizer_args = training_ctx.args.get("optimizer_args", {})
            optimizer = optimizer_type(self.parameters(), **optimizer_args)
        if isinstance(optimizer, dict):
            optimizer_type = getattr(torch.optim, optimizer["@type"])
            optimizer_args = {
                k: v for k, v in optimizer.items() if not k.startswith("@")
            }
            optimizer = optimizer_type(self.parameters(), **optimizer_args)
        training_ctx.optimizer = optimizer

    def _fit_configure_lr_scheduler(self):
        lr_scheduler = training_ctx.args.get("lr_scheduler")
        if isinstance(lr_scheduler, str):
            lr_scheduler_type = getattr(torch.optim.lr_scheduler, lr_scheduler)
            lr_scheduler_args = training_ctx.args.get("lr_schedule_args", {})
            lr_scheduler = lr_scheduler_type(
                training_ctx.optimizer, **lr_scheduler_args
            )
        if isinstance(lr_scheduler, dict):
            lr_scheduler_type = getattr(torch.optim.lr_scheduler, lr_scheduler["@type"])
            lr_scheduler_args = {
                k: v for k, v in lr_scheduler.items() if not k.startswith("@")
            }
            lr_scheduler = lr_scheduler_type(
                training_ctx.optimizer, **lr_scheduler_args
            )
        training_ctx.lr_scheduler = lr_scheduler

    def _fit_lr_scheduler_step(self, iter_type: str) -> None:
        scheduler = training_ctx.lr_scheduler
        if scheduler is None:
            return
        lr_scheduler_step = training_ctx.args.get("lr_scheduler_step")
        if lr_scheduler_step is None:
            scheduler.step()
            return
        if lr_scheduler_step.pop("on", "epoch") != iter_type:
            return
        scheduler.step(**lr_scheduler_step)

    def predict_iter(
        self,
        x: XY,
        y: Y = None,
        /,
        batch_size: int = 32,
        callbacks: Optional[Sequence[Callback]] = None,
    ) -> Generator[Tuple[int, RecordBatch, Z, LossDict[torch.Tensor]], None, None]:
        device = next(self.parameters()).device
        if not isinstance(callbacks, CallbackList):
            callbacks = CallbackList(callbacks)
        test_dataset = Dataset(x, y)
        test_dataloader = DataLoader(
            test_dataset,
            shuffle=False,
            batch_size=batch_size,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )
        callback_params = {}
        if callbacks.params is not None:
            callback_params.update(callbacks.params)
        callback_params.update({"pred_steps": len(test_dataloader)})
        callbacks.set_params(callback_params)
        self.eval()
        callbacks.on_predict_begin()
        batch_idx, total_loss_dict = 0, None
        for batch_idx, batch in enumerate(test_dataloader):
            callbacks.on_predict_batch_begin(batch_idx)
            if self.training:
                self.eval()
            batch = move_to_device(batch, device)
            with torch.no_grad():
                batch_output = self.predict_on_batch(batch)
                loss = self.compute_loss(batch, batch_output)
            # batch_output = move_to_device(detach(batch_output), "cpu")
            # batch_loss_dict = LossDict.from_loss(loss).detach()
            batch_loss_dict = LossDict.from_loss(loss)
            total_loss_dict = batch_loss_dict + total_loss_dict
            batch_logs = {}
            callbacks.on_predict_batch_end(batch_idx, logs=batch_logs)
            yield batch_idx, batch, batch_output, batch_loss_dict
            # clear memory cache
            del batch, batch_output, loss, batch_loss_dict
        n_batches = batch_idx + 1
        logs = {}
        if n_batches > 0 and total_loss_dict is not None:
            average_loss_dict = total_loss_dict / n_batches
            logs.update(average_loss_dict.item().to_dict())
        callbacks.on_predict_end(logs)

    def predict(
        self,
        x: XY,
        y: Y = None,
        /,
        batch_size: int = 32,
        callbacks: Optional[Sequence[Callback]] = None,
    ) -> Z:
        logits = None
        for _, _, batch_output, _ in self.predict_iter(
            x, y, batch_size=batch_size, callbacks=callbacks
        ):
            logits = concat((logits, batch_output["logits"]))
        if logits is None:
            msg = (
                f"no predictions found for the given input data of "
                f"type '{type(x).__name__}' and size {len(x)}"
            )
            raise ValueError(msg)
        return logits

    def evaluate(
        self,
        x: XY,
        y: Y = None,
        /,
        metrics: Union[Evaluator, Sequence[Metric], Dict[str, Metric], None] = None,
        batch_size: int = 32,
        include_loss: bool = True,
        return_dict: bool = False,
        prefix: str = "",
        callbacks: Optional[Sequence[Callback]] = None,
    ) -> Union[Dict[str, Any], Tuple[Any, ...]]:
        if not isinstance(metrics, Evaluator):
            metrics = Evaluator(metrics)
        metrics.reset()
        n_batches, total_loss_dict = 0, None
        for _, batch, output, batch_loss_dict in self.predict_iter(
            x,
            y,
            callbacks=callbacks,
            batch_size=batch_size,
        ):
            n_batches += 1
            total_loss_dict = batch_loss_dict + total_loss_dict
            eval_input = self.extract_eval_input(batch, output)
            metrics.update_state(**eval_input)
        average_loss_dict = {}
        if include_loss:
            if n_batches > 0 and total_loss_dict is not None:
                average_loss_dict = total_loss_dict / n_batches
                average_loss_dict = average_loss_dict.item().to_dict()
            # add prefix to loss keys
            average_loss_dict = {
                f"{prefix}{key}": value for key, value in average_loss_dict.items()
            }
        if return_dict:
            return {
                **average_loss_dict,
                **{
                    f"{prefix}{name}": value
                    for name, value in metrics.result(return_dict=True).items()
                },
            }
        return (
            *(v for _, v in sorted(average_loss_dict.items())),
            *metrics.result(),
        )

    def collate_fn(self, batch: Sequence[Record[XY, Y]]) -> RecordBatch:  # noqa: PLR6301
        batch = default_collate(batch)
        index = batch.pop("index")
        batch = batch["x"], batch.get("y", None)
        return RecordBatch(*batch, index=index)

    def compute_loss(self, batch: RecordBatch, output: Z) -> torch.Tensor:
        raise NotImplementedError

    def predict_on_batch(self, batch: RecordBatch) -> Z:
        raise NotImplementedError

    def extract_eval_input(self, batch: RecordBatch, output: Z) -> Dict[str, Any]:
        return {"y_true": batch.y, "y_pred": output, "y_score": output}
