from __future__ import annotations

import functools
import gc
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
from typing_extensions import ParamSpec, Self, Unpack

from tklearn.nn.callbacks import Callback, CallbackList
from tklearn.nn.loss import LossDict
from tklearn.nn.metrics import Evaluator, Metric
from tklearn.nn.utils.array import concat, detach, move_to_device
from tklearn.nn.utils.data import Dataset, Record, RecordBatch

P = ParamSpec("P")

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
    try:
        if torch.backends.mps.is_available():
            return "mps" if not torch.backends.mps.is_built() else "cpu"
    except AttributeError:
        pass
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
    callbacks: Optional[CallbackList]
    loss_args: Optional[Dict[str, Any]]


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

    @property
    def training_args(self) -> Dict[str, Any]:
        if not hasattr(self, "_training_args"):
            self._training_args = {}
        return self._training_args

    def fit(  # noqa: PLR0914
        self,
        x: XY,
        y: Y = None,
        /,
        **kwargs: Unpack[TrainingArgs],
    ) -> Self:
        self.training_args.clear()
        self.training_args.update(kwargs)
        batch_size = self.training_args.get("batch_size", 32)
        loss_args = self.training_args.get("loss_args", {})
        if isinstance(x, Dataset) and y is not None:
            msg = "y should be None when x is a Dataset"
            raise ValueError(msg)
        train_dataset = x if isinstance(x, Dataset) else Dataset(x, y)
        shuffle = self.training_args.get("shuffle", True)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self.collate_fn,
        )
        callbacks = self.training_args.get("callbacks", None)
        if not isinstance(callbacks, CallbackList):
            callbacks = CallbackList(callbacks)
        validation_data = self.training_args.get("validation_data")
        validation_batch_size = self.training_args.get(
            "validation_batch_size", batch_size
        )
        metrics = self.training_args.get("metrics")
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
            batch_idx, total_loss_dict = 0, None
            for batch_idx, batch in enumerate(train_dataloader):
                total_loss_dict = (
                    self._fit_on_batch(
                        batch=batch,
                        batch_idx=batch_idx,
                        callbacks=callbacks,
                        device=device,
                        optimizer=optimizer,
                        lr_scheduler=lr_scheduler,
                        loss_args=loss_args,
                    )
                    + total_loss_dict
                )
            self._fit_lr_scheduler_step(lr_scheduler, "epoch")
            n_batches = batch_idx + 1
            if n_batches > 0 and total_loss_dict is not None:
                epoch_logs = (total_loss_dict / n_batches).to_dict()
            else:
                epoch_logs = {}
            if evaluate is not None:
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
        callbacks: Optional[CallbackList] = None,
        batch_idx: Optional[int] = None,
        loss_args: Optional[Dict[str, Any]] = None,
    ) -> LossDict[float]:
        if loss_args is None:
            loss_args = {}
        if callbacks:
            callbacks.on_train_batch_begin(batch_idx)
        if not self.training:
            self.train()
        batch = move_to_device(batch, device)
        batch_output = self.predict_on_batch(batch)
        loss = self.compute_loss(batch, batch_output, **loss_args)
        loss_dict = LossDict.from_loss(loss)
        optimizer.zero_grad()
        loss_dict.loss.backward()
        self._fit_clip_grad_norm()
        optimizer.step()
        self._fit_lr_scheduler_step(lr_scheduler, "batch")
        batch_logs = {}
        if callbacks:
            callbacks.on_train_batch_end(
                batch_idx,
                logs=batch_logs,
            )
        return loss_dict.item()

    def _fit_clip_grad_norm(self) -> None:
        clip_grad_norm = self.training_args.get("clip_grad_norm")
        if clip_grad_norm is False or clip_grad_norm is None:
            return
        clip_grad_norm_args: dict = self.training_args.get("clip_grad_norm_args", {})
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
        torch.nn.utils.clip_grad_norm_(self.parameters(), **clip_grad_norm_args)

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
            lr_scheduler_type = getattr(torch.optim.lr_scheduler, lr_scheduler["$type"])
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
        callbacks: Optional[Sequence[Callback]] = None,
    ) -> Z:
        logits = None
        for _, _, batch_output, _ in self.predict_iter(
            x,
            y,
            batch_size=batch_size,
            callbacks=callbacks,
        ):
            logits = concat((logits, batch_output["logits"]))
        if logits is None:
            msg = (
                f"no predictions found for the given input data of "
                f"type '{type(x).__name__}' and size {len(x)}"
            )
            raise ValueError(msg)
        return logits

    def predict_iter(
        self,
        x: XY,
        y: Y = None,
        /,
        batch_size: int = 32,
        callbacks: Optional[Sequence[Callback]] = None,
        loss_args: Optional[Dict[str, Any]] = None,
    ) -> Generator[Tuple[int, RecordBatch, Z, LossDict[float]], None, None]:
        device = next(self.parameters()).device
        if not isinstance(callbacks, CallbackList):
            callbacks = CallbackList(callbacks)
        if isinstance(x, Dataset) and y is not None:
            msg = "y should be None when x is a Dataset"
            raise ValueError(msg)
        test_dataset = x if isinstance(x, Dataset) else Dataset(x, y)
        test_dataloader = DataLoader(
            test_dataset,
            shuffle=False,
            batch_size=batch_size,
            collate_fn=self.collate_fn,
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
            loss = self.compute_loss(batch, batch_output, **(loss_args or {}))
            batch_output = move_to_device(detach(batch_output), "cpu")
            batch_loss_dict = LossDict.from_loss(loss).item()
            total_loss_dict = batch_loss_dict + total_loss_dict
            batch_logs = {}
            callbacks.on_predict_batch_end(batch_idx, logs=batch_logs)
            yield batch_idx, batch, batch_output, batch_loss_dict
            # clear memory cache
            del batch, batch_output, loss, batch_loss_dict
        n_batches = batch_idx + 1
        logs = {}
        if n_batches > 0 and total_loss_dict is not None:
            average_loss_dict = (total_loss_dict / n_batches).to_dict()
            logs.update(average_loss_dict)
        callbacks.on_predict_end(logs)

    def predict_on_batch(self, batch: RecordBatch) -> Z:
        raise NotImplementedError

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
        loss_args: Optional[Dict[str, Any]] = None,
    ) -> Union[Dict[str, Any], Tuple[Any, ...]]:
        evaluator = metrics if isinstance(metrics, Evaluator) else Evaluator(metrics)
        evaluator.reset()
        n_batches, total_loss_dict = 0, None
        for _, batch, output, batch_loss_dict in self.predict_iter(
            x,
            y,
            callbacks=callbacks,
            batch_size=batch_size,
            loss_args=loss_args,
        ):
            n_batches += 1
            total_loss_dict = batch_loss_dict + total_loss_dict
            eval_input = self.extract_eval_input(batch, output)
            evaluator.update_state(**eval_input)
        average_loss_dict = {}
        if include_loss:
            if n_batches > 0 and total_loss_dict is not None:
                average_loss_dict = (total_loss_dict / n_batches).to_dict()
            # add prefix to loss keys
            average_loss_dict = {
                f"{prefix}{key}": value for key, value in average_loss_dict.items()
            }
        if return_dict:
            return {
                **average_loss_dict,
                **{
                    f"{prefix}{name}": value
                    for name, value in evaluator.result(return_dict=True).items()
                },
            }
        return (
            *(v for _, v in sorted(average_loss_dict.items())),
            *evaluator.result(),
        )

    def extract_eval_input(  # noqa: PLR6301
        self, batch: RecordBatch, output: Z
    ) -> Dict[str, Any]:
        return {"y_true": batch.y, "y_pred": output, "y_score": output}

    def compute_loss(
        self, batch: RecordBatch, output: Z, **kwargs: Any
    ) -> torch.Tensor:
        raise NotImplementedError

    def collate_fn(self, batch: Sequence[Record[XY, Y]]) -> RecordBatch:  # noqa: PLR6301
        batch = default_collate(batch)
        index = batch.pop("index")
        batch = batch["x"], batch.get("y", None)
        return RecordBatch(*batch, index=index)
