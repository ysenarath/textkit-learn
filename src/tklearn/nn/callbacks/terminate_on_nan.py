from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from opentelemetry import trace

from tklearn.logging import get_logger
from tklearn.nn.callbacks.base import Callback

if TYPE_CHECKING:
    from tklearn.nn.trainer import Trainer

__all__ = [
    "TerminateOnNaN",
]

logger = get_logger(__name__)


class TerminateOnNaN(Callback):
    """Stop training when a batch loss is NaN or infinite.

    Training ends after that batch, so its epoch is still evaluated and
    logged.
    """

    def on_train_batch_end(
        self, trainer: Trainer, batch: Any, logs: dict[str, float]
    ) -> None:
        loss = logs.get("loss")
        if loss is not None and not math.isfinite(loss):
            logger.warning(
                f"Step {trainer.global_step}: the loss is {loss}; "
                "terminating training"
            )
            trainer.should_stop = True
            trace.get_current_span().add_event(
                "terminate_on_nan",
                {"loss": loss, "step": trainer.global_step},
            )
