from typing import Any
from collections.abc import Mapping

import pyarrow as pa
import pandas as pd
import polars as pl

__all__ = [
    "create_table",
    "DEFAULT_BATCH_SIZE",
]

DEFAULT_BATCH_SIZE = 5_000_000


def create_table(data: Any) -> pa.Table:
    if isinstance(data, pa.Table):
        return data
    if isinstance(data, Mapping):
        return pa.Table.from_pydict(data)
    if isinstance(data, list):
        # list of dicts
        return pa.Table.from_pylist(data)
    if isinstance(data, pd.DataFrame):
        return pa.Table.from_pandas(data)
    if isinstance(data, pl.LazyFrame):
        return data.collect().to_arrow()
    if isinstance(data, pl.DataFrame):
        return data.to_arrow()
    raise ValueError(f"cannot create table from {type(data).__name__}")


def merge_tables(*tables: pa.Table) -> pa.Table:
    if len(tables) == 1:
        return tables[0]
    if len(tables) > 2:
        return merge_tables(tables[0], merge_tables(tables[1:]))
    x, y = tables
    return pa.Table.from_arrays(
        arrays=x.columns + y.columns,
        names=x.column_names + y.column_names,
    )
