from __future__ import annotations

import gc
import re
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

from tklearn.metrics import MetricBase, MetricState
from tklearn.nn.callbacks import Callback, CallbackList
from tklearn.nn.optim import (
    MultipleLRSchedulers,
    MultipleOptimizers,
    configure_lr_schedulers,
    configure_optimizers,
)
from tklearn.nn.utils.collections import TensorDict
from tklearn.nn.utils.data import Record, RecordBatch, TorchDataset, default_collate
from tklearn.utils.array import concat, detach, move_to_device

P = ParamSpec("P")

X, Y, Z = TypeVar("X"), TypeVar("Y"), TypeVar("Z")

XY = Union[X, Tuple[X, Y], None]

CollateFunctionType = Callable[[Sequence[Record[XY, Y]]], RecordBatch]


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
    metrics: Union[Sequence[MetricBase], Dict[str, MetricBase], None]
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
    epochs: Optional[int] = None
    steps: Optional[int] = None


@dataclass
class PredictionContext:
    args: Mapping[str, Any]
    steps: Optional[int] = None


@dataclass
class ModuleContext:
    training: Optional[TrainingContext] = None
    prediction: Optional[PredictionContext] = None


_global_cv = ContextVar("global_cv", default=ModuleContext())


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
    def context(self) -> ModuleContext:
        return _global_cv.get()

    def fit(self, x: XY, y: Y = None, /, **kwargs: Unpack[TrainingArgsDict]) -> Self:
        def run():
            ctx = ModuleContext(
                training=TrainingContext(args=kwargs),
            )
            token = _global_cv.set(ctx)
            try:
                return self._fit_with_context(x, y)
            finally:
                _global_cv.reset(token)

        return copy_context().run(run)

    def _fit_with_context(self, x: XY, y: Y = None) -> Self:
        training_args = self.context.training.args
        batch_size = training_args.get("batch_size", 32)
        train_dataset = TorchDataset(x, y)
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
        collate_fn = self.configure_collate_fn()
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
        validate = validation_data is not None and metrics is not None
        self.stop_training = False
        device = self.device
        steps = len(train_dataloader)
        epochs = training_args.get("epochs", 1)
        self.context.training.epochs = epochs
        self.context.training.steps = steps
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
            if validate:
                eval_results = self.evaluate(
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
    ) -> TensorDict[torch.Tensor]:
        batch = move_to_device(batch, device, non_blocking=True)
        if callbacks:
            callbacks.on_train_batch_begin(batch_idx)
        if not self.training:
            self.train()
        batch_output = self.predict_on_batch(batch)
        loss = self.compute_loss(batch, batch_output)
        batch_loss_dict = TensorDict(loss)
        self.context.training.optimizer.zero_grad()
        batch_loss_dict.backward()
        self._fit_clip_grad_norm()
        self.context.training.optimizer.step()
        self._fit_lr_scheduler_step("batch")
        batch_logs = {}
        if callbacks:
            callbacks.on_train_batch_end(
                batch_idx,
                logs=batch_logs,
            )
        return batch_loss_dict.detach()

    def _fit_clip_grad_norm(self) -> None:
        clip_grad_norm = self.context.training.args.get("clip_grad_norm")
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
        torch.nn.utils.clip_grad_norm_(self.parameters(), **clip_grad_norm)

    def _fit_configure_optimizers(self):
        optimizers = self.configure_optimizers()
        if isinstance(optimizers, (List, Tuple)):
            optimizers = MultipleOptimizers(*optimizers)
        self.context.training.optimizer = optimizers

    def _fit_configure_lr_schedulers(self):
        lr_schedulers = self.configure_lr_schedulers()
        if isinstance(lr_schedulers, (List, Tuple)):
            lr_schedulers = MultipleLRSchedulers(*lr_schedulers)
        self.context.training.lr_scheduler = lr_schedulers

    def _fit_lr_scheduler_step(self, iter_type: str) -> None:
        scheduler = self.context.training.lr_scheduler
        if scheduler is None:
            return
        lr_scheduler_step = self.context.training.args.get("lr_scheduler_step")
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
    ) -> Generator[Tuple[int, RecordBatch, Z, TensorDict[torch.Tensor]], None, None]:
        def run():
            ctx = ModuleContext(
                # keep existing training context
                training=self.context.training,
                # update prediction context
                prediction=PredictionContext(
                    args={
                        "batch_size": batch_size,
                        "callbacks": callbacks,
                        "sampler": sampler,
                        "batch_sampler": batch_sampler,
                        "collate_fn": collate_fn,
                    }
                ),
            )
            token = _global_cv.set(ctx)
            try:
                yield from self._predict_iter_with_context(x, y)
            finally:
                # reset global context
                _global_cv.reset(token)

        return copy_context().run(run)

    def _predict_iter_with_context(self, x: XY, y: Y = None) -> Generator:
        device = self.device
        test_dataset = TorchDataset(x, y)
        # batch_sampler option is mutually exclusive with batch_size, shuffle, sampler, and drop_last
        batch_size = self.context.prediction.args["batch_size"]
        batch_sampler = self.context.prediction.args.get("batch_sampler", None)
        sampler = self.context.prediction.args.get("sampler", None)
        if batch_sampler is not None:
            batch_size = 1
            sampler = None
        elif sampler is not None:
            batch_size = 1
        collate_fn = self.configure_collate_fn()
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
        callbacks = self.context.prediction.args.get("callbacks", None)
        if not isinstance(callbacks, CallbackList):
            callbacks = CallbackList(callbacks)
        if callbacks.params is not None:
            callback_params.update(callbacks.params)
        self.context.prediction.steps = len(test_dataloader)
        callback_params.update({
            "pred_steps": self.context.prediction.steps,
        })
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
            batch_loss_dict = TensorDict(loss)
            total_loss_dict = batch_loss_dict + total_loss_dict
            batch_logs = {}
            callbacks.on_predict_batch_end(batch_idx, logs=batch_logs)
            # print(batch, batch_output)
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
                f"type '{x.__class__.__name__}' and size {len(x)}"
            )
            raise ValueError(msg)
        return logits

    def evaluate(
        self,
        x: XY,
        y: Y = None,
        /,
        metrics: Union[Dict[str, MetricBase], List[MetricBase], None] = None,
        batch_size: int = 32,
        include_loss: bool = True,
        return_dict: bool = True,
        prefix: str = "",
        callbacks: Optional[Sequence[Callback]] = None,
        sampler: Optional[Sampler] = None,
        batch_sampler: Optional[Sampler] = None,
        collate_fn: Optional[CollateFunctionType] = None,
    ) -> Union[Dict[str, Any], Tuple[Any, ...]]:
        if not isinstance(metrics, MetricState):
            metrics = MetricState(metrics)
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
            metrics.update(**eval_input)
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
        raise NotImplementedError

    def compute_loss(self, batch: RecordBatch, output: Z) -> torch.Tensor:
        raise NotImplementedError

    def predict_on_batch(self, batch: RecordBatch) -> Z:
        raise NotImplementedError

    def extract_eval_input(self, batch: RecordBatch, output: Z) -> Dict[str, Any]:
        raise NotImplementedError

    def configure_optimizers(self) -> Union[Optimizer, Tuple[Optimizer]]:
        config = self.context.training.args.get("optimizer")
        return configure_optimizers(self, config)

    def configure_lr_schedulers(self) -> Union[LRScheduler, Tuple[LRScheduler], None]:
        config = self.context.training.args.get("lr_scheduler")
        return configure_lr_schedulers(self, config)

    def configure_collate_fn(self) -> CollateFunctionType:
        if self.context.prediction is None:
            collate_fn = self.context.training.args.get("collate_fn")
            if collate_fn is None:
                collate_fn = default_collate
        else:
            collate_fn = self.context.prediction.args.get("collate_fn")
            if collate_fn is None:
                collate_fn = default_collate
        return collate_fn

    def freeze_layers(
        self, layers: Optional[List[str]] = None, prefix: str = ""
    ) -> int:
        """
        Freeze layers in the model that match the given patterns.

        Parameters
        ----------
        layers : list of str, optional
            A list of layer names or patterns to freeze. Supports wildcards (*)
            and dot notation for nested layers. If None, no layers will be frozen.
        prefix : str, default=""
            An optional prefix to apply to all layer patterns.

        Returns
        -------
        int
            The number of parameters frozen.

        Raises
        ------
        ValueError
            If an invalid regex pattern is provided.

        Examples
        --------
        >>> model.freeze_layers(['encoder.*', 'encoder.layer.[0-8].*'])
        >>> model.freeze_layers(['layer_[1-3]'], prefix='transformer')

        Notes
        -----
        This method uses regular expressions to match layer names. Dots in layer
        names are treated as literal dots, while asterisks are treated as wildcards.
        """
        if not layers:
            return 0  # no layers to freeze
        # escape dots and convert asterisks to regex wildcards
        layers = [p.replace(".", r"\.").replace("*", ".*") for p in layers]
        pattern_regex = "|".join(layers)
        if prefix:
            pattern_regex = f"{prefix}\.({pattern_regex})"
        # compile regex pattern
        try:
            pattern = re.compile(f"^{pattern_regex}$")
        except re.error as e:
            raise ValueError(str(e))
        # freeze parameters that match the pattern
        frozen_params = 0
        for name, param in self.named_parameters():
            if not pattern.match(name):
                continue
            param.requires_grad = False
            frozen_params += param.numel()
        # return number of frozen parameters
        return frozen_params
