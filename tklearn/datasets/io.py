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
        self._table_buffer = []
        self._buffer = []

    def flush(self, force=False):
        if (len(self._buffer) >= self._batch_size) or (force and len(self._buffer) > 0):
            buffer_table = create_table(self._buffer)
            self._table_buffer.append(buffer_table)
            self._buffer = []
        if len(self._table_buffer) == 0:
            return
        # write to locked folder only
        path = self._path.with_suffix(".lock")
        # list of dicts
        base_dir = path / "data"
        if not base_dir.exists():
            base_dir.mkdir(parents=True, exist_ok=True)
        for table in self._table_buffer:
            uuid_hex = uuid.uuid4().hex
            name_template = f"part-{uuid_hex}-{{i}}.{self._format}"
            ds.write_dataset(
                table,  # from list of tables
                base_dir=base_dir,
                format=self._format,
                basename_template=name_template,
                existing_data_behavior="overwrite_or_ignore",
            )
        self._table_buffer = []

    def write(self, data: Union[dict, list, pa.Table]):
        if isinstance(data, dict):
            self._buffer.append(data)
        elif isinstance(data, list):
            self._buffer.extend(data)
        else:
            data = create_table(data)
            self._table_buffer.append(data)
        self.flush()

    def close(self):
        self.flush(force=True)

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
        if not base_dir.exists():
            base_dir.mkdir(parents=True, exist_ok=True)
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
