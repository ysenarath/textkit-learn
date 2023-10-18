from __future__ import annotations
import shutil
import weakref
from functools import partial, wraps
from pathlib import Path
import typing
from typing import Union

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
from pyarrow import compute as pc

from tklearn.datasets.io import DatasetWriter, DatasetReader
from tklearn.datasets.utils import create_table, merge_tables, DEFAULT_BATCH_SIZE
from tklearn.utils.cache import getcachedir, mkdtemp
from tklearn.utils.hash import hash
from tklearn.utils import logging


__all__ = [
    "Dataset",
]


logger = logging.get_logger(__name__)


class MapBatch(typing.Iterable):
    def __init__(self, func: partial, batched: bool = True):
        # creates a new dataset
        self.func = func
        # wheather output of a function is a batch or not
        self.batched = batched

    @property
    def hash(self):
        return hash(self.func)

    def iter(self):
        for batch in self.func():
            yield batch

    def __iter__(self):
        if self.batched:
            yield from self.iter()
        else:
            yield self.iter()

    def to_dataset(self) -> Dataset:
        path = getcachedir() / f"dataset-mapped-{self.hash}"
        if path.exists():
            # print("-(exist)->", path)
            return Dataset(path)
        # else:
        #     print("-(new)->", path)
        with DatasetWriter(path) as writer:
            for row in self.iter():
                if self.batched:
                    row = create_table(row)
                writer.write(row)
        return Dataset(path)


def batched(func) -> typing.Callable:
    @wraps(func)
    def map_batched(self: Dataset, *args, **kwargs):
        pf = partial(func, self, *args, **kwargs)
        return MapBatch(pf, batched=True).to_dataset()

    return map_batched


class DatasetPathMixin(object):
    def __init__(
        self,
        path: Union[str, Path, None] = None,
        format: str = "arrow",
    ):
        self._format = format
        # path none means its a virtual dataset
        self._finalizer = None
        if path is None:
            # create a temporary directory
            path = mkdtemp(prefix="dataset-")
            # attach temporary directory to the object lifecycle
            self._finalizer = weakref.finalize(
                self,
                DatasetPathMixin._shutil_rmtree,
                path,
            )
        path: Path = Path(path)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
        self._path: Path = path

    @classmethod
    def _shutil_rmtree(cls, path: Path):
        try:
            shutil.rmtree(path)
        except Exception as e:
            logger.warning(f"failed to remove {path}: {e}")

    @property
    def path(self) -> Path:
        return self._path

    @property
    def format(self) -> str:
        return self._format

    def __getstate__(self):
        return {
            "path": str(self.path),
            "format": self.format,
        }

    def __setstate__(self, state):
        self._path = Path(state["path"])
        self._format = state["format"]

    def open(self, mode) -> Union[DatasetReader, DatasetWriter]:
        if mode == "w":
            return DatasetWriter(self.path, format=self.format)
        if mode == "r":
            return DatasetReader(self.path, format=self.format)
        raise ValueError(f"invalid mode {mode}")


class Dataset(DatasetPathMixin):
    def __init__(
        self,
        path_or_data: typing.Union[str, Path, None] = None,
        format: str = "arrow",
    ):
        if isinstance(path_or_data, (str, Path)):
            data = None
            path = path_or_data
        else:
            data = path_or_data
            path = None
        super(Dataset, self).__init__(path=path, format=format)
        if data is not None:
            # pyarrow base already exsit
            # copy it to the path (likely a temp path)
            with self.open("w") as writer:
                writer.write(data)
        # load/update pyarrow base
        with self.open("r") as reader:
            self._pyarrow_base = reader.read()

    @property
    def schema(self) -> pa.Schema:
        return self._pyarrow_base.schema

    @batched
    def take(self, indices: list[int]) -> typing.Generator[pa.Table, None, None]:
        if not isinstance(indices, list):
            indices = list(indices)
        for idx_batch in range(0, len(indices), DEFAULT_BATCH_SIZE):
            idx_slice = indices[idx_batch : idx_batch + DEFAULT_BATCH_SIZE]  # noqa
            base_table = self._pyarrow_base.take(idx_slice)
            yield base_table

    def extend(
        self,
        data: typing.Union[
            pl.DataFrame,
            pd.DataFrame,
            pl.LazyFrame,
            typing.List[dict],
            typing.Dict[str, typing.Any],
            pa.Table,
        ],
    ) -> Dataset:
        with self.open("w") as writer:
            writer.write(data)
        with self.open("r") as reader:
            self._pyarrow_base = reader.read()
        return self

    def __getitem__(self, index) -> typing.Union[pa.Table, dict]:
        if self._pyarrow_base is None:
            raise IndexError("dataset index out of range")
        if isinstance(index, int):
            # return dict
            items = self._pyarrow_base.take([index]).to_pylist()
            return items[0]
        if isinstance(index, slice):
            # return table
            # convert slice to list
            start = index.start or 0
            stop = index.stop or len(self)
            if index.step:
                index = list(range(start, stop, index.step))
            else:
                index = list(range(start, stop))
            return self._pyarrow_base.take(index)
        else:
            # return table
            return self._pyarrow_base.take(index)

    def to_batches(
        self, batch_size: typing.Optional[int] = None
    ) -> typing.Generator[pa.Table, None, None]:
        if batch_size is None:
            batch_size = DEFAULT_BATCH_SIZE
        for batch in self._pyarrow_base.to_batches(
            batch_size=batch_size,
        ):
            yield pa.Table.from_batches([batch])

    def __iter__(self) -> typing.Generator[dict, None, None]:
        for table in self.to_batches():
            for row in table.to_pylist():
                yield row

    def __len__(self) -> int:
        return self._pyarrow_base.count_rows()

    @batched
    def _map_arrow(
        self,
        func: typing.Callable,
        verbose: bool = True,
        batched: bool = None,
        batch_size: typing.Optional[int] = None,
        batch_into: typing.Optional[type] = None,
        keep_columns: bool = False,
    ) -> typing.Generator[pa.Table, None, None]:
        total = len(self)
        pbar = None
        if verbose:
            pbar = logging.ProgressBar(desc="Map", total=total)
        if batched is None and batch_into is not None:
            batched = True
        elif batched and batch_into is None:
            batch_into = pd.DataFrame
        for table in self.to_batches(batch_size=batch_size):
            if batched:
                if batch_into is pd.DataFrame:
                    result = func(table.to_pandas())
                elif batch_into is pl.DataFrame:
                    result = func(pl.from_arrow(table))
                else:
                    result = func(table)
            else:
                result = [func(item) for item in table.to_pylist()]
            if not isinstance(result, pa.Table):
                # convert to table
                result = create_table(result)
            # result is always a table
            if keep_columns:
                yield merge_tables(table, result)
            else:
                yield result
            # update progress bar
            if pbar is not None:
                pbar.update(table.num_rows)
        if pbar:
            pbar.close()

    @batched
    def _map_polars(
        self, func: typing.Callable, verbose: bool = False
    ) -> typing.Generator[pa.Table, None, None]:
        # check if path exists
        input_lf = self.to_polars(lazy=True)
        lf = func(input_lf)
        if isinstance(lf, pl.DataFrame):
            lf = lf.lazy()
        if not isinstance(lf, pl.LazyFrame):
            table = create_table(lf)
            yield table
            return
        cf = lf.cache()
        height: int = cf.select(pl.count()).collect().row(0)[0]
        # lock the dataset until its finished written
        pbar = None
        if verbose:
            pbar = logging.ProgressBar(desc="Map", total=height)
        for i in range(0, height, DEFAULT_BATCH_SIZE):
            batch = cf.slice(i, i + DEFAULT_BATCH_SIZE).collect()
            tab = batch.to_arrow()  # type: pa.Table
            yield tab
            if pbar:
                pbar.update(tab.num_rows)
        if pbar:
            pbar.close()

    @typing.overload
    def map(
        self,
        func: typing.Callable,
        *,
        mode: str = "arrow",
        verbose: bool = True,
        batch_size: typing.Optional[int] = None,
        batch_into: typing.Optional[type] = None,
    ) -> Dataset:
        ...

    @typing.overload
    def map(
        self,
        func: typing.Callable,
        *,
        mode: str = "polars",
        verbose: bool = True,
    ) -> Dataset:
        ...

    def map(
        self,
        func: typing.Callable,
        *,
        mode: typing.Optional[str] = None,
        verbose: bool = True,
        **kwargs,
    ) -> Dataset:
        if mode == "polars":
            return self._map_polars(func, verbose=verbose)
        if (mode is None) or (mode in {"arrow", "batched"}):
            return self._map_arrow(func, verbose=verbose, **kwargs)
        raise ValueError(f"invalid mode {mode}")

    @batched
    def rename_columns(
        self,
        columns: typing.Union[typing.Dict[str, str], typing.List[str]],
        verbose: bool = True,
    ) -> typing.Generator[pa.Table, None, None]:
        # output is a virtual dataset
        pbar = None
        if verbose:
            total = len(self)
            pbar = logging.ProgressBar(desc="Rename", total=total)
        for table in self.to_batches():
            if isinstance(columns, dict):
                cols: list[str] = [columns.get(c, c) for c in self.schema.names]
            else:
                cols: list[str] = columns
            yield table.rename_columns(cols)
            if pbar:
                pbar.update(table.num_rows)

    def rename_column(self, original_column_name: str, new_column_name: str) -> Dataset:
        return self.rename_columns({original_column_name: new_column_name})

    @batched
    def remove_columns(
        self, columns: typing.Union[str, typing.List[str]], verbose: bool = True
    ) -> typing.Generator[pa.Table, None, None]:
        if isinstance(columns, str):
            columns = [columns]
        pbar = None
        if verbose:
            total = len(self)
            pbar = logging.ProgressBar(desc="Remove", total=total)
        col_indexes_to_remove = []
        for i, c in enumerate(self.schema.names):
            if c in columns:
                col_indexes_to_remove.append(i)
        for table in self.to_batches():
            for col in col_indexes_to_remove:
                table = table.remove_column(col)
            yield table
            if pbar:
                pbar.update(table.num_rows)

    @batched
    def filter(self, expr: pc.Expression) -> typing.Generator[pa.Table, None, None]:
        arrow_dset = self._pyarrow_base.filter(expr)
        # output is a virtual dataset
        for batch in arrow_dset.to_batches():
            yield pa.Table.from_batches([batch])

    def filter_by(self, **kwargs) -> Dataset:
        expr = None
        for key, value in kwargs.items():
            if isinstance(value, str):
                value = [value]
            expr_or = None
            for s in value:
                if expr_or is None:
                    expr_or = pc.field(key) == s
                else:
                    expr_or = expr_or | (pc.field(key) == s)
            if expr is None:
                expr = expr_or
            else:
                expr = expr & expr_or
        return self.filter(expr)

    @batched
    def select(
        self,
        columns: typing.Union[str, typing.List[str], typing.Dict[str, pc.Expression]],
        batch_size: typing.Optional[int] = None,
    ) -> typing.Generator[pa.Table, None, None]:
        if isinstance(columns, str):
            columns = [columns]
        scanner = self._pyarrow_base.scanner(
            columns=columns, batch_size=batch_size or DEFAULT_BATCH_SIZE
        )
        for batch in scanner.to_batches():
            yield pa.Table.from_batches([batch])

    def head(self, n=5) -> pa.Table:
        return self._pyarrow_base.head(n)

    def to_table(self) -> pa.Table:
        return self._pyarrow_base.to_table()

    def to_pandas(self) -> pd.DataFrame:
        return self.to_table().to_pandas()

    def to_polars(
        self,
        lazy=True,
    ) -> typing.Union[pl.DataFrame, pl.LazyFrame]:
        if lazy:
            return pl.scan_ipc(self.path / "data" / "*.arrow")
        raise NotImplementedError

    @classmethod
    def from_polars(
        cls,
        obj: typing.Union[pl.DataFrame, pl.LazyFrame],
        path: typing.Union[str, Path],
        format: str = "arrow",
        verbose: bool = True,
    ) -> Dataset:
        # check if path exists
        if isinstance(obj, pl.DataFrame):
            # create a new dataset
            obj = obj.lazy()
        if path is not None and Path(path).exists():
            raise ValueError(f"path {path} already exists")
        cdf = obj.cache()
        height: int = cdf.select(pl.count()).collect().row(0)[0]
        pbar = None
        if verbose:
            pbar = logging.ProgressBar(desc="Map", total=height)
        temp: Dataset = cls(path, format=format)
        for i in range(0, height, DEFAULT_BATCH_SIZE):
            batch = cdf.slice(i, i + DEFAULT_BATCH_SIZE).collect()
            tab = batch.to_arrow()
            temp.extend(tab)
            if pbar:
                pbar.update(tab.num_rows)
        if pbar:
            pbar.close()
        return temp

    def first(self) -> dict:
        return self[0]

    def last(self) -> dict:
        return self[len(self) - 1]

    def __repr__(self) -> str:
        return f"Dataset({self.path}, {self.format}, {len(self)})"

    def _repr_html_(self) -> str:
        table_html_str = self.head().to_pandas().to_html(notebook=True)
        return f"""<style>
            .tklearn-container {{
                border: 1px solid #ccc;
                padding: 10px;
                margin: 10px;
            }}
            </style>
            <div class="tklearn-container">
                <table style="width: 100%;">
                    <tr>
                        <th class="label">Path</th>
                        <th>Format</th>
                        <th>#Rows</th>
                        <th>#Columns</th>
                    </tr>
                    <tr>
                        <td>{self.path}</td>
                        <td>{self.format}</td>
                        <td>{len(self)}</td>
                        <td>{len(self.schema.names)}</td>
                    </tr>
                </table>
                <hr>
                <div>{table_html_str}</div>
            </div>"""
