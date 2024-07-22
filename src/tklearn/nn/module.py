from __future__ import annotations

import re
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import torch
from torch import nn
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import Sampler
from torch.utils.data.dataloader import DataLoader
from typing_extensions import TypeVar

from tklearn.metrics import MetricBase, MetricState
from tklearn.nn.callbacks import Callback, CallbackList
from tklearn.nn.utils.collections import TensorDict
from tklearn.nn.utils.data import Record, RecordBatch, TorchDataset, default_collate
from tklearn.utils.array import concat, detach, move_to_device

X, Y, Z = TypeVar("X"), TypeVar("Y"), TypeVar("Z")

XY = Union[X, Tuple[X, Y], None]

CollateFunctionType = Callable[[Sequence[Record[XY, Y]]], RecordBatch]

_tensor_or_tensors_type = Union[torch.Tensor, Iterable[torch.Tensor]]

_clip_grad_norm_type = Union[
    int, float, bool, Dict[str, Any], Callable[[_tensor_or_tensors_type], None]
]


class Module(nn.Module):
    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def compile(
        self,
        fullgraph: bool = False,
        dynamic: bool | None = None,
        backend: str | Callable[..., Any] = "inductor",
        mode: str | None = None,
        options: Dict[str, str | int | bool] | None = None,
        disable: bool = False,
    ) -> Any:
        return torch.compile(
            self,
            fullgraph=fullgraph,
            dynamic=dynamic,
            backend=backend,
            mode=mode,
            options=options,
            disable=disable,
        )

    def predict_on_batch(self, batch: RecordBatch) -> Any:
        raise NotImplementedError

    def compute_loss(
        self, batch: RecordBatch, output: Any, **kwargs
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        return None

    def prepare_metric_inputs(
        self, batch: RecordBatch, batch_output: Any
    ) -> Dict[str, Any]:
        return {}

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

    def predict_iter(
        self,
        x: X,
        y: Y = None,
        *,
        batch_size: int = 32,
        collate_fn: Optional[CollateFunctionType] = None,
        sampler: Optional[Sampler] = None,
        batch_sampler: Optional[Sampler] = None,
        callbacks: Optional[CallbackList] = None,
        loss: Optional[Callable] = None,
    ) -> Generator[Tuple[int, RecordBatch, Any, TensorDict[torch.Tensor]], None, None]:
        device = self.device
        test_dataset = TorchDataset(x, y)
        # batch_sampler option is mutually exclusive with batch_size, shuffle, sampler, and drop_last
        if batch_sampler is not None:
            batch_size = 1
            sampler = None
        elif sampler is not None:
            batch_size = 1
        if collate_fn is None:
            collate_fn = default_collate
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
        if not isinstance(callbacks, CallbackList):
            callbacks = CallbackList(callbacks)
        if callbacks.params is not None:
            callback_params.update(callbacks.params)
        pred_steps = len(test_dataloader)
        callback_params.update({"pred_steps": pred_steps})
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
                batch_loss = self.compute_loss(batch, batch_output, loss_func=loss)
            batch_output = detach(batch_output)
            batch_output = move_to_device(batch_output, "cpu")
            if batch_loss:
                batch_loss_dict = TensorDict(batch_loss)
                total_loss_dict = batch_loss_dict + total_loss_dict
            else:
                batch_loss_dict = None
            batch_logs = {}
            callbacks.on_predict_batch_end(batch_idx, logs=batch_logs)
            # yield tuple
            yield batch_idx, batch, batch_output, batch_loss_dict
            # clear memory cache
            del batch, batch_output, batch_loss, batch_loss_dict
        n_batches = batch_idx + 1
        logs = {}
        if n_batches > 0 and total_loss_dict is not None:
            average_loss_dict = total_loss_dict / n_batches
            logs.update(average_loss_dict.item().to_dict())
        callbacks.on_predict_end(logs)

    def predict(
        self,
        x: X,
        y: Y = None,
        *,
        batch_size: int = 32,
        collate_fn: Optional[CollateFunctionType] = None,
        sampler: Optional[Sampler] = None,
        batch_sampler: Optional[Sampler] = None,
        callbacks: Optional[CallbackList] = None,
    ) -> Z:
        if not isinstance(callbacks, CallbackList):
            callbacks = CallbackList(callbacks)
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
        metrics: Union[
            Dict[dict, MetricBase], List[MetricBase], MetricBase, None
        ] = None,
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
            metric_inputs = self.prepare_metric_inputs(batch, output)
            metrics.update(**metric_inputs)
        average_loss_dict = {}
        if include_loss:
            if n_batches > 0 and total_loss_dict is not None:
                average_loss_dict = total_loss_dict / n_batches
                average_loss_dict = average_loss_dict.item().to_dict()
            # add prefix to loss keys
            average_loss_dict = {
                f"{prefix}{key}": value for key, value in average_loss_dict.items()
            }
        results = metrics.result()
        return {
            **average_loss_dict,
            **{f"{prefix}{name}": value for name, value in results.items()},
        }

    def fit_on_batch(
        self,
        batch: RecordBatch,
        *,
        optimizer: Optimizer,
        loss: Optional[Callable] = None,
        clip_grad_norm: _clip_grad_norm_type = None,
        lr_scheduler: Optional[LRScheduler] = None,
        device: Union[str, torch.device, None] = None,
        callbacks: Optional[CallbackList] = None,
        batch_idx: Optional[int] = None,
    ) -> TensorDict[torch.Tensor]:
        if device is None:
            device = self.device
        if not isinstance(callbacks, CallbackList):
            callbacks = CallbackList(callbacks)
        batch = move_to_device(batch, device, non_blocking=True)
        callbacks.on_train_batch_begin(batch_idx)
        if not self.training:
            self.train()
        batch_output = self.predict_on_batch(batch)
        batch_loss = self.compute_loss(batch, batch_output, loss_func=loss)
        batch_loss = TensorDict(batch_loss)
        optimizer.zero_grad()
        batch_loss.backward()
        if clip_grad_norm:
            if isinstance(clip_grad_norm, (int, float, bool)):
                torch.nn.utils.clip_grad_norm_(
                    self.parameters(), max_norm=float(clip_grad_norm)
                )
            elif isinstance(clip_grad_norm, Mapping):
                torch.nn.utils.clip_grad_norm_(self.parameters(), **clip_grad_norm)
            else:
                clip_grad_norm(self.parameters())
        optimizer.step()
        if lr_scheduler:
            lr_scheduler.step()
        batch_logs = {}
        callbacks.on_train_batch_end(batch_idx, logs=batch_logs)
        return batch_loss.detach()

    def fit(
        self,
        x: XY,
        y: Y = None,
        batch_size: int = 32,
        epochs: int = 1,
        shuffle: bool = True,
        loss: Optional[Callable] = None,
        optimizer: Union[Optimizer, None] = None,
        lr_scheduler: Union[LRScheduler, None] = None,
        lr_scheduler_step: Optional[str] = None,
        clip_grad_norm: _clip_grad_norm_type = None,
        callbacks: Optional[CallbackList] = None,
        sampler: Optional[Sampler] = None,
        batch_sampler: Optional[Sampler] = None,
        metrics: Union[Dict[str, MetricBase], List[MetricBase], None] = None,
        validation_data: XY = None,
        validation_batch_size: Optional[int] = None,
        validation_sampler: Optional[Sampler] = None,
        validation_batch_sampler: Optional[Sampler] = None,
        collate_fn: Optional[CollateFunctionType] = None,
    ):
        train_dataset = TorchDataset(x, y)
        # batch_sampler option is mutually exclusive with batch_size, shuffle, sampler, and drop_last
        if batch_sampler is not None:
            batch_size = 1
            shuffle = None
            sampler = None
        elif sampler is not None:
            batch_size = 1
            shuffle = None
        # create train dataloader
        if collate_fn is None:
            collate_fn = default_collate
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn,
            pin_memory=True,
        )
        if not isinstance(callbacks, CallbackList):
            callbacks = CallbackList(callbacks)
        if validation_batch_size is None:
            validation_batch_size = batch_size
        validate = validation_data is not None and metrics is not None
        device = self.device
        steps = len(train_dataloader)
        params = {"batch_size": batch_size, "epochs": epochs, "steps": steps}
        self.train()
        callbacks.set_params(params)
        callbacks.set_model(self)
        callbacks.on_train_begin()
        epoch_logs = {}
        batch_lr_scheduler = None
        if lr_scheduler_step == "batch":
            batch_lr_scheduler = lr_scheduler
        for epoch_idx in range(epochs):
            callbacks.on_epoch_begin(epoch_idx)
            total_loss_dict, batch_idx = None, 0
            for batch_idx, batch in enumerate(train_dataloader):
                batch_loss_dict = self.fit_on_batch(
                    batch=batch,
                    optimizer=optimizer,
                    loss=loss,
                    clip_grad_norm=clip_grad_norm,
                    lr_scheduler=batch_lr_scheduler,
                    device=device,
                    callbacks=callbacks,
                    batch_idx=batch_idx,
                )
                total_loss_dict = batch_loss_dict + total_loss_dict
            if lr_scheduler_step == "epoch":
                lr_scheduler.step()
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
            if getattr(self, "stop_training", False):
                break
        callbacks.on_train_end(epoch_logs)
        return self
