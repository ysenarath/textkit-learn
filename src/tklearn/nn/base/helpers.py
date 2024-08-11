from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Generic,
    Iterable,
    List,
    Mapping,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

import torch
from torch.utils.data import DataLoader

from tklearn.metrics import MetricBase, MetricState
from tklearn.nn.base.module import Module
from tklearn.nn.callbacks.base import Callback, CallbackList
from tklearn.nn.loss import LossDict
from tklearn.nn.optim import LRScheduler, LRSchedulerConfig, Optimizer
from tklearn.utils.array import concat, move_to_device

_tensor_or_tensors_type = Union[torch.Tensor, Iterable[torch.Tensor]]

ClipGradNormType = Union[
    int, float, bool, Dict[str, Any], Callable[[_tensor_or_tensors_type], None]
]


ModelInput, ModelOutput = TypeVar("BatchInput"), TypeVar("BatchOutput")  # noqa: E741
LossLike = Union[torch.Tensor, Mapping[str, torch.Tensor], LossDict, None]
LossFunctionType = Callable[[ModelInput, ModelOutput], LossLike]
PostprocessorFunctionType = Callable[[ModelInput, ModelOutput], Dict[str, Any]]


class CallbacksPropertyMixin:
    @property
    def callbacks(self) -> CallbackList:
        return self._callbacks

    @callbacks.setter
    def callbacks(self, value: Union[CallbackList, Iterable[Callback], None]) -> None:
        if isinstance(value, Callback):
            value = [value]
        self._callbacks = CallbackList(value)


class Predictor(CallbacksPropertyMixin, Generic[ModelInput, ModelOutput]):
    def __init__(
        self,
        model: Module[ModelInput, ModelOutput],
        dataloader: DataLoader,
        loss: Optional[Callable[[ModelInput, ModelOutput], LossLike]] = None,
        callbacks: Union[CallbackList, Iterable[Callback], None] = None,
    ) -> None:
        self.model = model
        self.dataloader = dataloader
        self.callbacks = callbacks
        self.loss = loss

    @torch.no_grad()
    def iter_batches(
        self,
    ) -> Generator[Tuple[int, ModelInput, ModelOutput, LossDict], None, None]:
        if self.model.training:
            self.model.eval()
        # set the callback params
        callback_params = {}
        if self.callbacks.params is not None:
            callback_params.update(self.callbacks.params)
        callback_params.update({"pred_steps": len(self.dataloader)})
        self.callbacks.set_params(callback_params)
        self.callbacks.set_model(self.model)
        # start the prediction
        self.callbacks.on_predict_begin()
        dataloader_idx = None
        for batch_idx, batch in enumerate(self.dataloader):
            batch = move_to_device(batch, self.model.device, non_blocking=True)
            self.callbacks.on_predict_batch_begin(batch_idx)
            output = self.model.predict_step(
                batch, batch_idx=batch_idx, dataloader_idx=dataloader_idx
            )
            if self.loss is None:
                try:
                    batch_loss = self.model.compute_loss(batch, output)
                except NotImplementedError:
                    batch_loss = None
            else:
                batch_loss = self.loss(batch, output)
            batch_loss = LossDict(batch_loss)
            batch_logs = {}
            self.callbacks.on_predict_batch_end(batch_idx, logs=batch_logs)
            yield batch_idx, batch, output, batch_loss
        self.callbacks.on_predict_end()

    def predict(self) -> torch.Tensor:
        # change to eval mode
        self.model.eval()
        logits = None
        for batch_idx, batch, output, batch_loss in self.iter_batches():
            logits = concat((logits, output["logits"]))
        if logits is None:
            msg = "no predictions found for the given input data"
            raise ValueError(msg)
        return logits


class Evaluator(CallbacksPropertyMixin, Generic[ModelInput, ModelOutput]):
    def __init__(
        self,
        model: Module,
        dataloader: DataLoader,
        metrics: Union[Dict[str, MetricBase], Iterable[MetricBase], str, None] = None,
        # only used if metrics MetricBase based
        loss: Optional[LossFunctionType] = None,
        include_loss: bool = True,
        postprocessor: Optional[PostprocessorFunctionType] = None,
        prefix: str = "",
        callbacks: Union[CallbackList, Iterable[Callback], None] = None,
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
    def _valiate_or_test(self, mode: str) -> List[Dict[str, Any]]:
        if self.model.training:
            self.model.eval()
        if mode in ("validate", "valid", "val"):
            mode = "validate"
        if mode not in ("validate", "test"):
            # valid values for mode: validate, valid, val, test
            msg = f"mode must be one of 'validate' or 'test', got {mode} instead"
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
        self.callbacks.on_predict_begin()
        for batch_idx, batch in enumerate(self.dataloader):
            batch = move_to_device(batch, self.model.device, non_blocking=True)
            self.callbacks.on_predict_batch_begin(batch_idx)
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
            self.callbacks.on_predict_batch_end(batch_idx, logs=batch_logs)
            outputs.append(output)
        self.callbacks.on_predict_end()
        return outputs

    def validate(self) -> List[Dict[str, Any]]:
        return self._valiate_or_test("validate")

    def test(self) -> List[Dict[str, Any]]:
        return self._valiate_or_test("test")

    def evaluate(self) -> Dict[str, Any]:
        if self.metrics == "validate":
            return self.validate()
        elif self.metrics == "test":
            return self.test()
        metric_state = self._create_metric_state()
        predictor: Predictor[ModelInput, ModelOutput] = Predictor(
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
                f"{self.prefix}{key}": value for key, value in average_loss.items()
            }
        results = metric_state.result()
        if not isinstance(results, Mapping) and isinstance(results, Iterable):
            results = {
                f"{self.prefix}metric[{i}]": results[i] for i in range(len(results))
            }
        return {
            **average_loss,
            **{f"{self.prefix}{name}": value for name, value in results.items()},
        }


class Trainer(CallbacksPropertyMixin, Generic[ModelInput, ModelOutput]):
    def __init__(
        self,
        model: Module[ModelInput, ModelOutput],
        dataloader: DataLoader,
        optimizer: Optimizer,
        loss: Optional[LossFunctionType] = None,
        epochs: int = 1,
        lr_scheduler: Union[LRScheduler, LRSchedulerConfig, None] = None,
        clip_grad_norm: ClipGradNormType = None,
        callbacks: Union[CallbackList, Iterable[Callback], None] = None,
        evaluator: Optional[Evaluator] = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.epochs = epochs
        self.lr_scheduler = lr_scheduler
        self.clip_grad_norm = clip_grad_norm
        self.callbacks = callbacks
        self.loss = loss
        self.evaluator = evaluator

    def _training_step(
        self,
        batch: ModelInput,
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        device: Union[torch.device, str, None] = None,
    ) -> LossDict[torch.Tensor]:
        if device is None:
            device = self.model.device
        batch = move_to_device(batch, device, non_blocking=True)
        self.callbacks.on_train_batch_begin(batch_idx)
        if not self.model.training:
            self.model.train()
        if self.loss is None:
            # if loss is not defined, try to use the training_step method
            # if that is not implemented, try to use the predict_step method
            # with compute_loss
            try:
                batch_loss = self.model.training_step(
                    batch, batch_idx=batch_idx, dataloader_idx=dataloader_idx
                )
            except NotImplementedError:
                try:
                    batch_output = self.model.predict_step(
                        batch, batch_idx=batch_idx, dataloader_idx=dataloader_idx
                    )
                    batch_loss = self.model.compute_loss(batch, batch_output)
                except NotImplementedError:
                    batch_loss = None
        else:
            # if the loss is provided externally, use that instead with the
            # predict_step method
            batch_output = self.model.predict_step(
                batch, batch_idx=batch_idx, dataloader_idx=dataloader_idx
            )
            batch_loss = self.loss(batch, batch_output)
        batch_loss = LossDict(batch_loss)
        self.optimizer.zero_grad()
        batch_loss.backward()
        if self.clip_grad_norm:
            if isinstance(self.clip_grad_norm, (int, float, bool)):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=float(self.clip_grad_norm)
                )
            elif isinstance(self.clip_grad_norm, Mapping):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), **self.clip_grad_norm
                )
            else:
                self.clip_grad_norm(self.model.parameters())
        self.optimizer.step()
        if self.lr_scheduler:
            self.lr_scheduler.step()
        batch_logs = {}
        self.callbacks.on_train_batch_end(batch_idx, logs=batch_logs)
        return batch_loss.detach()

    def train(self) -> None:
        device = self.model.device
        move_to_device(self.model, device, non_blocking=True)
        steps = len(self.dataloader)
        self.model.train()
        params = {
            "batch_size": self.dataloader.batch_size,
            "epochs": self.epochs,
            "steps": steps,
        }
        self.callbacks.set_params(params)
        self.callbacks.set_model(self.model)
        self.callbacks.on_train_begin()
        epoch_logs = {}
        for epoch_idx in range(self.epochs):
            self.callbacks.on_epoch_begin(epoch_idx)
            total_loss, batch_idx = None, 0
            dataloader_idx = None
            for batch_idx, batch in enumerate(self.dataloader):
                batch_loss = self._training_step(
                    batch,
                    batch_idx=batch_idx,
                    dataloader_idx=dataloader_idx,
                    device=device,
                )
                total_loss = batch_loss + total_loss
            epoch_logs = {}
            if total_loss is not None:
                # average the loss (dict)
                epoch_logs = (total_loss / (batch_idx + 1)).item().to_dict()
            if self.evaluator:
                eval_results = self.evaluator.evaluate()
                epoch_logs.update(eval_results)
            self.callbacks.on_epoch_end(epoch_idx, logs=epoch_logs)
            if getattr(self.model, "stop_training", False):
                break
        self.callbacks.on_train_end(epoch_logs)
