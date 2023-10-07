from __future__ import annotations
from typing import Union

from pathlib import Path
import os
import uuid

import pyarrow as pa
import pyarrow.dataset as ds

from tklearn.datasets.utils import DEFAULT_BATCH_SIZE, create_table

__all__ = [
    "DatasetWriter",
]


class DatasetWriter:
    """A class to collect data batch-wise from an iterable
    and write it to a dataset.

    Examples
    --------
    >>> with DatasetWriter(dataset) as writer:
    ...     for row in data:
    ...         writer.write(row)
    """

    def __init__(
        self,
        path: Union[Path, str],
        format: str = "arrow",
        batch_size: int = DEFAULT_BATCH_SIZE,
        verbose: bool = True,
    ):
        self._path = Path(path)
        self._format = format
        self._batch_size = batch_size
        self._verbose = verbose
        self._buffer = []

    def flush(self):
        # write to locked folder only
        path = self._path.with_suffix(".lock")
        if len(self._buffer) == 0:
            return
        # list of dicts
        uuid_hex = uuid.uuid4().hex
        name_template = f"part-{uuid_hex}-{{i}}.{self._format}"
        base_dir = path / "data"
        if not base_dir.exists():
            base_dir.mkdir(parents=True, exist_ok=True)
        ds.write_dataset(
            create_table(self._buffer),
            base_dir=base_dir,
            format=self._format,
            basename_template=name_template,
            existing_data_behavior="overwrite_or_ignore",
        )
        self._buffer = []

    def write(self, data: Union[dict, list, pa.Table]):
        if isinstance(data, dict):
            self._buffer.append(data)
        elif isinstance(data, list):
            self._buffer.extend(data)
        else:
            data = create_table(data).to_pylist()
            self._buffer.extend(data)
        if len(self._buffer) >= self._batch_size:
            self.flush()

    def __enter__(self):
        """Enter the runtime context related to this object."""
        # if exists rename to .lock
        if self._path.exists():
            os.rename(self._path, self._path.with_suffix(".lock"))
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # rename .lock to original
        self.close()
        locked_path = self._path.with_suffix(".lock")
        if locked_path.exists():
            os.rename(locked_path, self._path)

    def close(self):
        self.flush()


class DatasetReader(object):
    def __init__(
        self,
        path: Union[Path, str],
        format: str = "arrow",
    ) -> None:
        self.path = Path(path)
        self.format = format

    def read(self) -> ds.Dataset:
        base_dir = self.path / "data"
        return ds.dataset(
            str(base_dir),
            format=self.format,
        )

    def close(self):
        pass

    def __enter__(self):
        """Enter the runtime context related to this object."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
