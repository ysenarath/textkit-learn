from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from opentelemetry import trace

from tklearn.logging import get_logger
from tklearn.nn.callbacks.base import Callback

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
        self, batch: Any, logs: Mapping[str, float]
    ) -> None:
        loss = logs.get("loss")
        if loss is not None and not math.isfinite(loss):
            logger.warning(
                f"Step {self.trainer.global_step}: the loss is {loss}; "
                "terminating training"
            )
            self.trainer.should_stop = True
            trace.get_current_span().add_event(
                "terminate_on_nan",
                {"loss": loss, "step": self.trainer.global_step},
            )
