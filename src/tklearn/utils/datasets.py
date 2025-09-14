from __future__ import annotations

import functools
import gc
import logging
import tempfile
from collections.abc import Callable, Hashable
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator, Union

import pandas as pd
import torch
from datasets import (
    Dataset,
    DatasetDict,
    IterableDataset,
    IterableDatasetDict,
    concatenate_datasets,
)
from datasets import load_dataset as hf_load_dataset
from datasets.utils.logging import disable_progress_bar, enable_progress_bar
from tqdm import auto as tqdm

from tklearn.config import config

T_BI = dict[str, list[Any]]
T_BO = Union[dict[str, list[Any]], list[dict[str, Any]], pd.DataFrame]
T_I = dict[str, Any]
T_O = dict[str, Any]

__all__ = [
    "DatasetMapper",
    "load_dataset",
    "islice",
    "map_dataset",
    "GroupBy",
]


@contextmanager
def without_progress_bar():
    logging.getLogger("datasets").setLevel(logging.ERROR)
    disable_progress_bar()
    yield
    enable_progress_bar()
    logging.getLogger("datasets").setLevel(logging.INFO)


class DatasetMapper:
    def __init__(
        self,
        func: Callable[[T_BI], T_BO],
        *,
        batched: bool = True,
        temp_dir: Path | str | None = None,
    ):
        if isinstance(temp_dir, str):
            temp_dir = Path(temp_dir)
        elif temp_dir is None:
            # e.g., ~/.cache/tklearn/temp
            temp_dir = Path(config.temp_dir)
        self._temp_dir = temp_dir
        self.func = func
        self.batched = batched

    def __call__(self, *args, **kwargs) -> Any:
        return self.func(*args, **kwargs)

    @property
    def temp_dir(self) -> Path:
        if not self._temp_dir.exists():
            self._temp_dir.mkdir(parents=True)
        return self._temp_dir

    def map(
        self,
        dataset: Dataset | DatasetDict | T_BO,
        batch_size: int = 8,
        func_kwargs: dict | None = None,
        verbose: bool = False,
    ) -> Dataset:
        if isinstance(dataset, pd.DataFrame):
            dataset = Dataset.from_pandas(dataset)
        elif isinstance(dataset, dict) and not isinstance(
            dataset, DatasetDict
        ):
            dataset = Dataset.from_dict(dataset)
        elif isinstance(dataset, list):
            dataset = Dataset.from_list(dataset)
        if self.batched:
            self_map_dataset = self._map_batched_dataset
        else:
            msg = "batched=False is not supported yet"
            raise NotImplementedError(msg)
        if isinstance(dataset, Dataset):
            return self_map_dataset(
                dataset, batch_size, func_kwargs=func_kwargs, verbose=verbose
            )
        elif isinstance(dataset, DatasetDict):
            out = {}
            for key, ds in dataset.items():
                out[key] = self_map_dataset(
                    ds,
                    batch_size,
                    func_kwargs=func_kwargs,
                    verbose=verbose,
                )
            return DatasetDict(out)
        msg = f"expected dataset to be of type Dataset or DatasetDict, got {type(dataset)}"
        raise ValueError(msg)

    def _map_batched_dataset(
        self,
        dataset: Dataset,
        batch_size: int,
        func_kwargs: dict | None = None,
        verbose: bool = False,
    ) -> Dataset:
        if func_kwargs is None:
            func_kwargs = {}
        nlen = len(dataset)
        output = None
        with tempfile.TemporaryDirectory(dir=self.temp_dir) as tmp_dir:
            outpaths = []
            for i in tqdm.trange(0, nlen, batch_size, disable=not verbose):
                batch = dataset[i : i + batch_size]
                out = self.func(batch, **func_kwargs)
                if isinstance(out, dict):  # dict[col->list[any]]
                    out = pd.DataFrame.from_dict(out)
                elif isinstance(out, list):  # list[dict[col->any]]
                    out = pd.DataFrame.from_records(out)
                if not isinstance(out, pd.DataFrame):
                    msg = f"expected output to be of type dict or list, got {type(out)}"
                    raise ValueError(msg)
                outpath = Path(tmp_dir) / f"data-{i}.parquet"
                out.to_parquet(outpath, index=False)
                outpaths.append(outpath)
            output = Dataset.from_parquet(list(map(str, outpaths)))
        return output


@functools.wraps(hf_load_dataset)
def load_dataset(
    *args, **kwargs
) -> DatasetDict | Dataset | IterableDatasetDict | IterableDataset:
    return hf_load_dataset(*args, **kwargs)


def islice(
    dataset: Dataset | DatasetDict, *args, **kwargs
) -> Generator[dict, None, None]:
    split = kwargs.get("split", None)
    if split is None:
        n = dataset.num_rows
    else:
        n = dataset[split].num_rows
    for i in range(*args):
        if i >= n:
            break
        j = n + i if i < 0 else i
        if split is None:
            yield dataset[j]
        else:
            yield dataset[split][j]


def map_dataset(
    dataset: Dataset | DatasetDict | T_BO,
    func: Callable[[T_BI], T_BO],
    batched: bool = True,
    batch_size: int = 8,
    func_kwargs: dict | None = None,
    verbose: bool = False,
    *,
    temp_dir: Path | str | None = None,
) -> Dataset:
    mapper = DatasetMapper(func, batched=batched, temp_dir=temp_dir)
    return mapper.map(
        dataset,
        batch_size=batch_size,
        verbose=verbose,
        func_kwargs=func_kwargs,
    )


def create_groups_indices(
    keys: list[Hashable], idxs: list[int], groups: dict[Hashable, list[int]]
):
    for key, i in zip(keys, idxs):
        groups[key].append(i)


class GroupBy:
    def __init__(
        self,
        dataset: Dataset,
        by: str,
        batch_size: int = config.dataset_batch_size,
        verbose: int = 0,
    ):
        self._ds = dataset
        self._by = by
        self.verbose = verbose
        self.batch_size: int = batch_size
        self.__post_init__()

    def __post_init__(self):
        groups = {key: [] for key in self._ds.unique(self._by)}
        self._ds.map(
            create_groups_indices,
            with_indices=True,
            input_columns=self._by,
            fn_kwargs={"groups": groups},
            batched=True,
        )
        self._groups = {key: indices for key, indices in groups.items()}
        self._column_names = self._ds.column_names

    def agg(self, func: Callable, **kwargs) -> Dataset:
        group_items = list(self._groups.items())

        if self.verbose > 0:
            group_items = tqdm.tqdm(
                group_items,
                total=len(group_items),
                desc="Aggregating groups",
            )

        current_batch = []
        current_item_count = 0

        res = None

        for group_key, indices in group_items:
            group_size = len(indices)

            # If adding this group would exceed the batch size, process current batch
            if (
                current_batch
                and current_item_count + group_size > self.batch_size
            ):
                ds = _process_batch(
                    dataset=self._ds,
                    batch=current_batch,
                    by=self._by,
                    func=func,
                    fn_kwargs=kwargs,
                )
                res = concatenate_datasets([res, ds]) if res else ds

                # Reset for next batch
                current_batch = []
                current_item_count = 0
                # garbage collect
                gc.collect()
                torch.cuda.empty_cache()

                if self.verbose > 0:
                    log_mem_usage()

            # Add current group to batch
            current_batch.append((group_key, indices))
            current_item_count += group_size

        # Process any remaining groups in the final batch
        if current_batch:
            ds = _process_batch(
                dataset=self._ds,
                batch=current_batch,
                by=self._by,
                func=func,
                fn_kwargs=kwargs,
            )
            res = concatenate_datasets([res, ds]) if res else ds

        return ds


def _process_batch(
    dataset: Dataset,
    batch: list[tuple],
    by: str,
    func: Callable,
    fn_kwargs: dict,
) -> Dataset:
    """Process a batch of groups and return the aggregated dataset."""
    all_indices = sum((group_indices for _, group_indices in batch), [])
    # print(f"Processing batch with {len(batch)} groups, total {len(all_indices)} items")
    with without_progress_bar():
        return Dataset.from_generator(
            _apply_func_to_groups,
            gen_kwargs={
                "data": dataset.select(all_indices),
                "by": by,
                "func": func,
                "fn_kwargs": fn_kwargs,
            },
            num_proc=1,
        )


def _apply_func_to_groups(
    data: Dataset, by: str, func: Callable, fn_kwargs: dict
) -> Generator[dict, None, None]:
    items = data.to_list()
    for _, group in pd.DataFrame.from_records(items).groupby(by):
        result = func(group, **fn_kwargs)
        if isinstance(result, pd.DataFrame):
            for record in result.to_dict(orient="records"):
                yield record
        elif isinstance(result, dict):
            yield result
        elif isinstance(result, list):
            yield from result
        else:
            msg = f"expected output to be of type dict or pd.DataFrame, got {type(result)}"
            raise ValueError(msg)


def log_mem_usage():
    try:
        import psutil

        process = psutil.Process()
        mem_info = process.memory_info()
        rss_in_mb = mem_info.rss / (1024 * 1024)
        print(f"Current memory usage: {rss_in_mb:.2f} MB")
    except ImportError:
        pass
