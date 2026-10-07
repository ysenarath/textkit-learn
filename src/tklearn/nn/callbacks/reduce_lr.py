from __future__ import annotations

from typing import TYPE_CHECKING, Any

from opentelemetry import trace

from tklearn.logging import get_logger
from tklearn.nn.callbacks._monitor import Mode, MonitorCallback

if TYPE_CHECKING:
    from tklearn.nn.trainer import Trainer

__all__ = [
    "ReduceLROnPlateau",
]

logger = get_logger(__name__)

# smallest learning rate change that is applied, as in torch's
# ReduceLROnPlateau
_EPS = 1e-8


class ReduceLROnPlateau(MonitorCallback):
    """Reduce the learning rate when a monitored value stops improving.

    After `patience` epochs without improvement, every parameter group's
    learning rate is multiplied by `factor`, down to `min_lr`.

    Parameters
    ----------
    monitor : str, default="valid_loss"
        Key of the watched value in the epoch logs.
    factor : float, default=0.1
        Multiplier of the learning rate, in ``(0, 1)``.
    patience : int, default=10
        Epochs without improvement after which the learning rate drops.
    mode : {"auto", "min", "max"}, default="auto"
        Whether lower or higher values are better; ``"auto"`` infers it
        from the key.
    min_delta : float, default=1e-4
        Smallest change that counts as an improvement.
    cooldown : int, default=0
        Epochs to wait after a reduction before counting epochs without
        improvement again.
    min_lr : float, default=0
        Lower bound of the learning rate.
    verbose : bool, default=False
        Log every reduction.

    Notes
    -----
    The callback sets the learning rate itself, so the trainer must not
    have an ``lr_scheduler``, which would overwrite it.
    """

    def __init__(
        self,
        monitor: str = "valid_loss",
        *,
        factor: float = 0.1,
        patience: int = 10,
        mode: Mode = "auto",
        min_delta: float = 1e-4,
        cooldown: int = 0,
        min_lr: float = 0.0,
        verbose: bool = False,
    ) -> None:
        super().__init__(monitor, mode, min_delta)
        if not 0 < factor < 1:
            msg = f"factor must be in (0, 1), got {factor}"
            raise ValueError(msg)
        self.factor = factor
        self.patience = patience
        self.cooldown = cooldown
        self.min_lr = min_lr
        self.verbose = verbose
        self._reset()

    def _reset(self) -> None:
        self.best = self._initial_best()
        self.wait = 0
        self.cooldown_counter = 0

    def on_train_begin(self, trainer: Trainer) -> None:
        if trainer.lr_scheduler is not None:
            msg = (
                "ReduceLROnPlateau sets the learning rate, which the "
                "trainer's lr_scheduler would overwrite; use one of them"
            )
            raise ValueError(msg)
        self._reset()

    def on_epoch_end(self, trainer: Trainer, logs: dict[str, Any]) -> None:
        current = self._get_monitor_value(logs)
        if current is None:
            return
        improved = self._is_improvement(current, self.best)
        if improved:
            self.best = current
        if self.cooldown_counter > 0:
            # unlike Keras, which counts the last epoch of the cooldown
            self.cooldown_counter -= 1
        elif improved:
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience and self._reduce(trainer):
                self.cooldown_counter = self.cooldown
                self.wait = 0

    def _reduce(self, trainer: Trainer) -> bool:
        """Reduce the learning rates; whether any changed."""
        reduced = False
        for i, group in enumerate(trainer.optimizer.param_groups):
            old_lr = float(group["lr"])
            new_lr = max(old_lr * self.factor, self.min_lr)
            if old_lr - new_lr > _EPS:
                group["lr"] = new_lr
                reduced = True
                trace.get_current_span().add_event(
                    "reduce_lr",
                    {"group": i, "old_lr": old_lr, "new_lr": new_lr},
                )
                if self.verbose and trainer.accelerator.is_main_process:
                    logger.info(
                        f"Epoch {trainer.epoch + 1}: reduced the learning "
                        f"rate of group {i} from {old_lr:.4g} to {new_lr:.4g}"
                    )
        return reduced
