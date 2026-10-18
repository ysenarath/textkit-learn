from __future__ import annotations

from collections.abc import Mapping

import torch
from accelerate import Accelerator
from accelerate.utils import gather_object

from tklearn.nn._outputs import BatchLogs

__all__ = [
    "RunningMean",
]


class RunningMean:
    """Means of logged values over the batches of every process.

    `fit` averages the training logs of each epoch with one; a callback
    can average the logs of `Callback.on_train_batch_end` over a window
    of batches the same way. Values are summed on the device, so `update`
    does not wait for it; `compute` does, once.

    Parameters
    ----------
    accelerator : Accelerator
        The accelerator of the session, e.g. ``self.session.accelerator``
        in a callback, whose device holds the sums and whose processes
        `compute` reduces over.

    Examples
    --------
    >>> means = RunningMean(self.session.accelerator)
    >>> means.update({"loss": torch.tensor(0.5)})
    >>> means.update(logs)  # the logs of on_train_batch_end
    >>> means.compute()
    {'loss': 0.45}
    """

    def __init__(self, accelerator: Accelerator) -> None:
        self._accelerator = accelerator
        self._device = accelerator.device
        # MPS has no float64
        self._dtype = (
            torch.float32 if self._device.type == "mps" else torch.float64
        )
        self._totals: dict[str, torch.Tensor] = {}
        self._counts: dict[str, int] = {}

    def update(self, logs: Mapping[str, torch.Tensor | float]) -> None:
        """Add the values logged by a batch.

        Tensors are added on the device. The logs of a training batch are
        read without converting them to floats.
        """
        if isinstance(logs, BatchLogs):
            logs = logs._values
        for key, value in logs.items():
            if key not in self._totals:
                self._totals[key] = torch.zeros(
                    (), dtype=self._dtype, device=self._device
                )
            self._totals[key] += value
            self._counts[key] = self._counts.get(key, 0) + 1

    def compute(self) -> dict[str, float]:
        """The mean of each value over the batches that logged it.

        On several processes this is a collective: every process must call
        it at the same point, or the others hang. Processes may log
        different keys; each process gets the means of all of them.
        """
        keys = list(self._totals)
        multiprocess = self._accelerator.num_processes > 1
        if multiprocess:
            # processes must reduce tensors of the same keys in the same
            # order; gather_object orders them by rank, then by first
            # appearance
            gathered = gather_object([keys])
            keys = list(dict.fromkeys(k for ks in gathered for k in ks))
        if not keys:
            return {}
        zero = torch.zeros((), dtype=self._dtype, device=self._device)
        totals = torch.stack([self._totals.get(k, zero) for k in keys])
        counts = torch.tensor(
            [self._counts.get(k, 0) for k in keys],
            dtype=self._dtype,
            device=self._device,
        )
        values = torch.cat([totals, counts])
        if multiprocess:
            values = self._accelerator.reduce(values, reduction="sum")
        values = values.tolist()
        n = len(keys)
        return {
            k: total / count
            for k, total, count in zip(keys, values[:n], values[n:])
            if count
        }
