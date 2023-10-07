import copy
from typing import Any, Optional, Union, TYPE_CHECKING
from pathlib import Path

import numpy as np
import torch
import transformers

from tklearn.core.callbacks import Callback, CallbackList
from tklearn.core.tracking import Run, Experiment
from tklearn.exceptions import EarlyStoppingException
from tklearn.utils.logging import Progbar

if TYPE_CHECKING:
    from optuna.trial import Trial
else:
    Trial = Any

__all__ = [
    "TrainerCallback",
    "TrainerCallbackList",
    "ProgbarLogger",
    "ModelCheckpoint",
    "TrackingCallback",
]


class TrainerCallback(Callback):
    def __init__(self) -> None:
        super().__init__()
        self.trainer: Any = None

    def set_trainer(self, trainer):
        self.trainer = trainer

    # on_(train|test|predict)_begin(self, logs=None)
    def on_train_begin(self, logs=None):
        pass

    def on_test_begin(self, logs=None):
        pass

    def on_predict_begin(self, logs=None):
        pass

    # on_(train|test|predict)_end(self, logs=None)
    def on_train_end(self, logs=None):
        pass

    def on_test_end(self, logs=None):
        pass

    def on_predict_end(self, logs=None):
        pass

    # on_(train|test|predict)_batch_begin(self, batch, logs=None)
    def on_train_batch_begin(self, batch: int, logs=None):
        pass

    def on_test_batch_begin(self, batch: int, logs=None):
        pass

    def on_predict_batch_begin(self, batch: int, logs=None):
        pass

    # on_(train|test|predict)_batch_end(self, batch, logs=None)
    def on_train_batch_end(self, batch: int, logs=None):
        pass

    def on_test_batch_end(self, batch: int, logs=None):
        pass

    def on_predict_batch_end(self, batch: int, logs=None):
        pass

    # on_epoch_begin(self, epoch, logs=None)
    def on_epoch_begin(self, epoch: int, logs=None):
        pass

    # on_epoch_end(self, epoch, logs=None)
    def on_epoch_end(self, epoch: int, logs=None):
        pass


class TrainerCallbackList(CallbackList, TrainerCallback):
    pass


class ProgbarLogger(TrainerCallback):
    def __init__(self) -> None:
        super().__init__()
        self.train_progbar: Optional[Progbar] = None
        self.predict_progbar: Optional[Progbar] = None
        self.train_steps = None
        self.predict_steps = None
        self.verbose = 1
        self.epochs = 1
        self._postfix = {}

    def set_params(self, params):
        if params is None:
            return
        if "epochs" in params:
            self.epochs = int(params["epochs"])
        if "steps" in params:
            self.train_steps = int(params["steps"])
        if "verbose" in params:
            self.verbose = int(params["verbose"])
        if "predict_steps" in params:
            self.predict_steps = int(params["predict_steps"])

    def on_train_begin(self, logs=None):
        if self.verbose and self.train_steps is not None:
            self.train_progbar = Progbar(
                total=self.train_steps,
                desc="Train",
            )
            self.train_progbar.set_postfix(
                {
                    "epoch": "0",
                }
            )
        else:
            self.train_progbar = None

    def on_epoch_begin(self, epoch: int, logs=None):
        if self.train_progbar is None:
            return
        self._postfix["epoch"] = f"{epoch}/{self.epochs}"
        self.train_progbar.set_postfix(self._postfix)

    def on_train_batch_end(self, batch: int, logs=None):
        if self.train_progbar is None:
            return
        self.train_progbar.update(1)

    def on_epoch_end(self, epoch: int, logs: Optional[dict] = None):
        if self.train_progbar is None or logs is None or len(logs) == 0:
            return
        # ic(logs)
        self._postfix.update(logs)
        self.train_progbar.set_postfix(self._postfix)

    def on_train_end(self, logs=None):
        if self.train_progbar is None:
            return
        self.train_progbar.close()

    def on_predict_begin(self, logs=None):
        if self.verbose and self.predict_steps is not None:
            self.predict_progbar = Progbar(
                total=self.predict_steps,
                desc="Predict",
                leave=self.train_progbar is None,
            )
        else:
            self.predict_progbar = None

    def on_predict_batch_end(self, batch: int, logs=None):
        if self.predict_progbar is None:
            return
        self.predict_progbar.update(1)

    def on_predict_end(self, logs=None):
        if self.predict_progbar is None:
            return
        self.predict_progbar.close()


class History(TrainerCallback):
    def __init__(self) -> None:
        super().__init__()
        self.history = []

    def on_epoch_end(self, epoch: int, logs=None):
        self.history.append(logs)


class ModelCheckpoint(TrainerCallback):
    def __init__(
        self,
        filepath: Union[str, Path],
        # monitor: str = 'val_loss',
        verbose: int = 0,
        # save_best_only: bool = False,
        # save_weights_only: bool = False,
        # mode: str = 'auto',
        # save_freq='epoch',
        # options=None,
        # initial_value_threshold=None,
        **kwargs,
    ):
        self.filepath: Path = Path(filepath)
        self.verbose = verbose

    def on_epoch_end(self, epoch: int, logs=None):
        model = self.trainer.model
        if model is None:
            return
        if not logs:
            logs = {}
        if isinstance(model, transformers.PreTrainedModel):
            model_path = self.filepath / f"epoch-{epoch:0004d}"
            model.save_pretrained(model_path)
        elif isinstance(model, torch.nn.Module):
            torch.save(model.state_dict(), self.filepath / f"epoch-{epoch:0004d}")
        else:
            raise TypeError(f"unsupported model type: {type(model).__name__}")


class EarlyStopping(TrainerCallback):
    def __init__(
        self,
        monitor: str = "val_loss",
        min_delta: float = 0,
        patience: int = 0,
        verbose: int = 0,
        mode: str = "auto",
        baseline: Optional[float] = None,
        restore_best_weights: bool = False,
        start_from_epoch: int = 0,
    ) -> None:
        super().__init__()
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.verbose = verbose
        self.mode = mode
        self.baseline = baseline
        self.restore_best_weights = restore_best_weights
        self.start_from_epoch = start_from_epoch
        # {min, max, auto}
        self.wait = 0
        self.stopped_epoch = 0
        self.best = np.Inf if self.monitor_op == np.less else -np.Inf
        self.best_weights = None
        self.best_epoch = 0

    @property
    def mode(self) -> str:
        return self._mode

    @mode.setter
    def mode(self, mode):
        if mode not in ["auto", "min", "max"]:
            raise ValueError(
                f"mode {mode} is unknown, " 'expected one of ("auto", "min", "max")'
            )
        self._mode = mode
        if mode == "min":
            self.monitor_op = np.less
        elif mode == "max":
            self.monitor_op = np.greater
        else:
            if (
                self.monitor.endswith("acc")
                or self.monitor.endswith("accuracy")
                or self.monitor.endswith("auc")
            ):
                self.monitor_op = np.greater
            else:
                self.monitor_op = np.less
        if self.monitor_op == np.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1

    def on_train_begin(self, logs=None):
        self.wait = 0
        self.stopped_epoch = 0
        self.best = np.Inf if self.monitor_op == np.less else -np.Inf
        self.best_weights = None
        self.best_epoch = 0

    def on_epoch_end(self, epoch: int, logs=None):
        current = self.get_monitor_value(logs)
        if current is None or epoch < self.start_from_epoch:
            # If no monitor value exists or still in initial warm-up stage.
            return
        if self.restore_best_weights and self.best_weights is None:
            # Restore the weights after first epoch if no progress is ever made.
            self.best_weights = copy.deepcopy(self.trainer.model)
        self.wait += 1
        if self._is_improvement(current, self.best):
            self.best = current
            self.best_epoch = epoch
            if self.restore_best_weights:
                self.best_weights = copy.deepcopy(self.trainer.model)
            # Only restart wait if we beat both the baseline and our previous
            # best.
            if self.baseline is None or self._is_improvement(current, self.baseline):
                self.wait = 0
            return
        # Only check after the first epoch.
        if self.wait >= self.patience and epoch > 0:
            self.stopped_epoch = epoch
            if self.restore_best_weights and self.best_weights is not None:
                if self.verbose > 0:
                    pass
                self.trainer.set_model(self.best_weights)
            raise EarlyStoppingException

    def get_monitor_value(self, logs):
        logs = logs or {}
        monitor_value = logs.get(self.monitor)
        if monitor_value is None:
            pass
        return monitor_value

    def _is_improvement(self, monitor_value, reference_value):
        return self.monitor_op(monitor_value - self.min_delta, reference_value)


class TrackingCallback(TrainerCallback):
    def __init__(
        self,
        run_or_experiment: Union[Run, Experiment],
        nested: bool = False,
    ) -> None:
        super().__init__()
        self.nested = nested
        if not isinstance(run_or_experiment, Run):
            run_or_experiment = run_or_experiment.start_run()
        self.run_or_experiment = run_or_experiment
        self.run: Run = None  # type: ignore

    def on_train_begin(self, logs=None):
        if self.run:
            return
        run = self.run_or_experiment
        if self.nested:
            run = run.start_run()
        self.run = run

    def on_epoch_end(self, epoch: int, logs: Optional[dict] = None):
        if logs is None:
            logs = {}
        self.run.log_metrics(logs, step=epoch)


class ModelSelectionCallback(TrainerCallback):
    def __init__(
        self,
        trial: Trial,
        monitor: str = "val_loss",
    ) -> None:
        self.trial = trial
        self.monitor = monitor

    def on_epoch_end(self, epoch: int, logs: Optional[dict] = None):
        if logs is None:
            logs = {}
        if self.monitor not in logs:
            return
        intermediate_value = logs[self.monitor]
        self.trial.report(intermediate_value, epoch)
