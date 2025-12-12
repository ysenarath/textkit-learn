from __future__ import annotations

from pathlib import Path
from typing import Any

import diskcache
import orjson
import torch
from diskcache import UNKNOWN


def default(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy()
    raise TypeError


class JSONDisk(diskcache.Disk):
    def __init__(
        self, directory: Path | str, json_option: int | None = None, **kwargs
    ):
        self.json_option = json_option or (
            orjson.OPT_NON_STR_KEYS
            | orjson.OPT_NAIVE_UTC
            | orjson.OPT_SERIALIZE_NUMPY
        )
        super().__init__(directory, **kwargs)

    def put(self, key):
        data = orjson.dumps(key, default=default, option=self.json_option)
        return super().put(data)

    def get(self, key, raw):
        data = super().get(key, raw)
        return orjson.loads(data)

    def store(self, value, read, key=UNKNOWN):
        if not read:
            value = orjson.dumps(
                value, default=default, option=self.json_option
            )
        return super().store(value, read, key=key)

    def fetch(self, mode, filename, value, read):
        data = super().fetch(mode, filename, value, read)
        if not read:
            data = orjson.loads(data)
        return data
