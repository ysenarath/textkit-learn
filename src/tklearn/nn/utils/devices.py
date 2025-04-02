from __future__ import annotations

import torch

__all__ = [
    "get_device",
]


def get_device(device: str | None = None) -> torch.device:
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            device = "mps"
        else:
            device = "cpu"
    if isinstance(device, str):
        return torch.device(device)
    if isinstance(device, int):
        return torch.device("cuda", device)
    if isinstance(device, torch.device):
        return device
    msg = f"device '{device}' is not supported"
    raise ValueError(msg)
