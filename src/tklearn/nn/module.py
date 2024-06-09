from __future__ import annotations

import gc
from collections.abc import Mapping
from contextvars import ContextVar, copy_context
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Generic,
    List,
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
from torch.utils.data.sampler import Sampler
from typing_extensions import ParamSpec, Self, Unpack
from werkzeug.local import LocalProxy

from octoflow.utils import func
from tklearn.nn.callbacks import Callback, CallbackList
from tklearn.nn.loss import LossAccumulator
from tklearn.nn.metrics import Evaluator, Metric
from tklearn.nn.optim import (
    MultipleLRSchedulers,
    MultipleOptimizers,
    configure_lr_schedulers,
    configure_optimizers,
)
from tklearn.nn.utils.array import concat, detach, move_to_device
from tklearn.nn.utils.data import Dataset, Record, RecordBatch, default_collate

P = ParamSpec("P")

X, Y, Z = TypeVar("X"), TypeVar("Y"), TypeVar("Z")

XY = Union[X, Tuple[X, Y], None]

CollateFunctionType = Callable[[Sequence[Record[XY, Y]]], RecordBatch]

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
    optimizer: Union[Optimizer, Dict[str, Any], str]
    lr_scheduler: Union[LRScheduler, Dict[str, Any], str, None]
    lr_scheduler_step: Optional[str]
    clip_grad_norm: Union[int, float, bool, Dict[str, Any], None]
    callbacks: Optional[CallbackList]
    sampler: Optional[Sampler]
    batch_sampler: Optional[Sampler]
    validation_sampler: Optional[Sampler]
    validation_batch_sampler: Optional[Sampler]
    collate_fn: Optional[CollateFunctionType]


@dataclass
class TrainingContext:
    args: Mapping[str, Any]
    optimizer: Optimizer = None
    lr_scheduler: Optional[LRScheduler] = None
    batch_size: Optional[int] = None
    epochs: Optional[int] = None
    steps: Optional[int] = None


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

    @property
    def training_ctx(self) -> TrainingContext:
        return training_ctx

    def _fit_with_context(self, x: XY, y: Y = None) -> Self:
        training_args = training_ctx.args
        batch_size = training_args.get("batch_size", 32)
        train_dataset = Dataset(x, y)
        shuffle = training_args.get("shuffle", True)
        # batch_sampler option is mutually exclusive with batch_size, shuffle, sampler, and drop_last
        sampler = training_args.get("sampler", None)
        batch_sampler = training_args.get("batch_sampler", None)
        if batch_sampler is not None:
            batch_size = 1
            shuffle = None
            sampler = None
        elif sampler is not None:
            batch_size = 1
            shuffle = None
        collate_fn = training_args.get("collate_fn", default_collate)
        # batch_sampler option is mutually exclusive with batch_size, shuffle, sampler, and drop_last
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn,
            pin_memory=True,
        )
        callbacks = training_args.get("callbacks", None)
        if not isinstance(callbacks, CallbackList):
            callbacks = CallbackList(callbacks)
        validation_data = training_args.get("validation_data")
        validation_batch_size = training_args.get("validation_batch_size", batch_size)
        validation_sampler = training_args.get("validation_sampler", None)
        validation_batch_sampler = training_args.get("validation_batch_sampler", None)
        metrics = training_args.get("metrics")
        evaluate = None
        if validation_data is not None and metrics is not None:
            evaluate = func.bind(
                self.evaluate,
                validation_data,
                metrics=metrics,
                prefix="valid_",
                callbacks=callbacks,
                batch_size=validation_batch_size,
                return_dict=True,
                sampler=validation_sampler,
                batch_sampler=validation_batch_sampler,
                collate_fn=collate_fn,
            )
        self.stop_training = False
        device = self.device
        steps = len(train_dataloader)
        epochs = training_args.get("epochs", 1)
        training_ctx.batch_size = batch_size
        training_ctx.epochs = epochs
        training_ctx.steps = steps
        params = {"batch_size": batch_size, "epochs": epochs, "steps": steps}
        self.train()
        self._fit_configure_optimizers()
        self._fit_configure_lr_schedulers()
        callbacks.set_params(params)
        callbacks.set_model(self)
        callbacks.on_train_begin()
        epoch_logs = {}
        for epoch_idx in range(epochs):
            callbacks.on_epoch_begin(epoch_idx)
            total_loss_dict, batch_idx = None, 0
            for batch_idx, batch in enumerate(train_dataloader):
                batch_loss_dict = self._fit_on_batch(
                    batch=batch,
                    batch_idx=batch_idx,
                    callbacks=callbacks,
                    device=device,
                )
                total_loss_dict = batch_loss_dict + total_loss_dict
            self._fit_lr_scheduler_step("epoch")
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
    ) -> LossAccumulator[torch.Tensor]:
        batch = move_to_device(batch, device, non_blocking=True)
        if callbacks:
            callbacks.on_train_batch_begin(batch_idx)
        if not self.training:
            self.train()
        batch_output = self.predict_on_batch(batch)
        loss = self.compute_loss(batch, batch_output)
        batch_loss_dict = LossAccumulator.from_loss(loss)
        training_ctx.optimizer.zero_grad()
        batch_loss_dict.backward()
        self._fit_clip_grad_norm()
        training_ctx.optimizer.step()
        self._fit_lr_scheduler_step("batch")
        batch_logs = {}
        if callbacks:
            callbacks.on_train_batch_end(
                batch_idx,
                logs=batch_logs,
            )
        return batch_loss_dict.detach()

    def _fit_clip_grad_norm(self) -> None:
        clip_grad_norm = training_ctx.args.get("clip_grad_norm")
        if clip_grad_norm is False or clip_grad_norm is None:
            # do not clip gradients
            return
        if isinstance(clip_grad_norm, (int, float, bool)):
            clip_grad_norm = {
                "max_norm": float(clip_grad_norm),
            }
        elif not isinstance(clip_grad_norm, dict):
            msg = (
                "clip_grad_norm should be a boolean, int, float, or dict, but got "
                f"'{clip_grad_norm.__class__.__name__}'"
            )
            raise ValueError(msg)
        torch.nn.utils.clip_grad_norm_(
            self.parameters(),
            **clip_grad_norm,
        )

    def _fit_configure_optimizers(self):
        optimizers = self.configure_optimizers()
        if isinstance(optimizers, (List, Tuple)):
            optimizers = MultipleOptimizers(*optimizers)
        training_ctx.optimizer = optimizers

    def _fit_configure_lr_schedulers(self):
        lr_schedulers = self.configure_lr_schedulers()
        if isinstance(lr_schedulers, (List, Tuple)):
            lr_schedulers = MultipleLRSchedulers(*lr_schedulers)
        training_ctx.lr_scheduler = lr_schedulers

    def _fit_lr_scheduler_step(self, iter_type: str) -> None:
        scheduler = training_ctx.lr_scheduler
        if scheduler is None:
            return
        lr_scheduler_step = training_ctx.args.get("lr_scheduler_step")
        if lr_scheduler_step is None:
            lr_scheduler_step = "epoch"
        if lr_scheduler_step != iter_type:
            return
        lr_scheduler_step_args = {}
        scheduler.step(**lr_scheduler_step_args)

    def predict_iter(
        self,
        x: XY,
        y: Y = None,
        /,
        batch_size: int = 32,
        callbacks: Optional[Sequence[Callback]] = None,
        sampler: Optional[Sampler] = None,
        batch_sampler: Optional[Sampler] = None,
        collate_fn: Optional[CollateFunctionType] = None,
    ) -> Generator[
        Tuple[int, RecordBatch, Z, LossAccumulator[torch.Tensor]], None, None
    ]:
        if collate_fn is None:
            collate_fn = default_collate
        device = next(self.parameters()).device
        if not isinstance(callbacks, CallbackList):
            callbacks = CallbackList(callbacks)
        test_dataset = Dataset(x, y)
        # batch_sampler option is mutually exclusive with batch_size, shuffle, sampler, and drop_last
        if batch_sampler is not None:
            batch_size = 1
            sampler = None
        elif sampler is not None:
            batch_size = 1
        test_dataloader = DataLoader(
            test_dataset,
            shuffle=False,
            batch_size=batch_size,
            sampler=sampler,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn,
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
            batch_output = detach(batch_output)
            batch_output = move_to_device(batch_output, "cpu")
            batch_loss_dict = LossAccumulator.from_loss(loss)
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
        sampler: Optional[Sampler] = None,
        batch_sampler: Optional[Sampler] = None,
        collate_fn: Optional[CollateFunctionType] = None,
    ) -> Z:
        logits = None
        for _, _, batch_output, _ in self.predict_iter(
            x,
            y,
            batch_size=batch_size,
            callbacks=callbacks,
            sampler=sampler,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn,
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
        sampler: Optional[Sampler] = None,
        batch_sampler: Optional[Sampler] = None,
        collate_fn: Optional[CollateFunctionType] = None,
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
            sampler=sampler,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn,
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

    def compute_loss(self, batch: RecordBatch, output: Z) -> torch.Tensor:
        raise NotImplementedError

    def predict_on_batch(self, batch: RecordBatch) -> Z:
        raise NotImplementedError

    def extract_eval_input(self, batch: RecordBatch, output: Z) -> Dict[str, Any]:
        return {"y_true": batch.y, "y_pred": output, "y_score": output}

    def configure_optimizers(self) -> Union[Optimizer, Tuple[Optimizer]]:
        config = training_ctx.args.get("optimizer")
        return configure_optimizers(self, config)

    def configure_lr_schedulers(self) -> Union[LRScheduler, Tuple[LRScheduler], None]:
        config = training_ctx.args.get("lr_scheduler")
        return configure_lr_schedulers(self, config)
