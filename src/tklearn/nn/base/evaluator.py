from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Callable, Generic, TypeVar, Union

import torch
from torch.utils.data import DataLoader

from tklearn.metrics import MetricBase, MetricState
from tklearn.nn.base.module import Module
from tklearn.nn.base.predictor import Predictor
from tklearn.nn.callbacks.base import Callback, CallbackList, CallbacksMixin
from tklearn.nn.loss import LossDict
from tklearn.utils.array import move_to_device

K = TypeVar("K")
V = TypeVar("V")
# L = torch.Tensor | Mapping[str, torch.Tensor] | LossDict | None
L = Union[torch.Tensor, Mapping[str, torch.Tensor], LossDict, None]


class Evaluator(CallbacksMixin, Generic[K, V]):
    def __init__(
        self,
        model: Module,
        dataloader: DataLoader,
        metrics: dict[str, MetricBase]
        | Iterable[MetricBase]
        | str
        | None = None,
        # only used if metrics MetricBase based
        loss: Callable[[K, V], L] | None = None,
        include_loss: bool = True,
        postprocessor: Callable[[K, V], dict[str, Any]] | None = None,
        prefix: str = "",
        callbacks: CallbackList | Iterable[Callback] | None = None,
    ) -> None:
        self.model = model
        self.dataloader = dataloader
        self.metrics = metrics
        self.loss = loss
        self.include_loss = include_loss
        self.postprocessor = postprocessor
        self.prefix = prefix
        self.callbacks = callbacks

    def _create_metric_state(self) -> MetricState:
        metrics = MetricState(self.metrics)
        metrics.reset()
        return metrics

    @torch.no_grad()
    def _validate_or_test(self, mode: str) -> list[dict[str, Any]]:
        if self.model.training:
            self.model.eval()
        if mode in ("validate", "valid", "val"):
            mode = "validate"
        if mode not in ("validate", "test"):
            # valid values for mode: validate, valid, val, test
            msg = (
                f"mode must be one of 'validate' or 'test', got {mode} instead"
            )
            raise ValueError(msg)
        if mode == "test":
            evaluation_step = self.model.test_step
        else:
            evaluation_step = self.model.validation_step
        dataloader_idx = None
        outputs = []
        # set the callback params
        callback_params = {}
        if self.callbacks.params is not None:
            callback_params.update(self.callbacks.params)
        callback_params.update({"pred_steps": len(self.dataloader)})
        self.callbacks.set_params(callback_params)
        self.callbacks.set_model(self.model)
        self.callbacks.on_test_begin()
        for batch_idx, batch in enumerate(self.dataloader):
            batch = move_to_device(batch, self.model.device, non_blocking=True)
            self.callbacks.on_test_batch_begin(batch_idx)
            output = evaluation_step(
                batch, batch_idx=batch_idx, dataloader_idx=dataloader_idx
            )
            batch_logs = {}
            if output is not None:
                if isinstance(output, torch.Tensor):
                    # the loss tensor is the only output
                    output = {"loss": output}
                if not isinstance(output, Mapping):
                    msg = (
                        f"output of '{mode}_step' must be a mapping or tensor, "
                        f"got {output.__class__.__name__} instead"
                    )
                    raise ValueError(msg)
                batch_logs.update(output)
            self.callbacks.on_test_batch_end(batch_idx, logs=batch_logs)
            outputs.append(output)
        self.callbacks.on_test_end()
        return outputs

    def validate(self) -> list[dict[str, Any]]:
        return self._validate_or_test("validate")

    def test(self) -> list[dict[str, Any]]:
        return self._validate_or_test("test")

    def evaluate(self) -> dict[str, Any]:
        if self.metrics == "validate":
            return self.validate()
        elif self.metrics == "test":
            return self.test()
        metric_state = self._create_metric_state()
        predictor: Predictor[K, V] = Predictor(
            model=self.model,
            dataloader=self.dataloader,
            loss=self.loss,
            callbacks=self.callbacks,
        )
        total_loss, n_batches = None, 0
        for _, batch, output, batch_loss in predictor.iter_batches():
            n_batches += 1
            total_loss = batch_loss + total_loss
            if self.postprocessor is None:
                metric_inputs = self.model.compute_metric_inputs(batch, output)
            else:
                metric_inputs = self.postprocessor(batch, output)
            metric_state.update(**metric_inputs)
        average_loss = {}
        if self.include_loss:
            if total_loss is not None:
                average_loss = (total_loss / n_batches).item().to_dict()
            # add prefix to loss keys
            average_loss = {
                f"{self.prefix}{key}": value
                for key, value in average_loss.items()
            }
        results = metric_state.result()
        if not isinstance(results, Mapping) and isinstance(results, Iterable):
            results = {
                f"{self.prefix}metric[{i}]": results[i]
                for i in range(len(results))
            }
        return {
            **average_loss,
            **{
                f"{self.prefix}{name}": value
                for name, value in results.items()
            },
        }
