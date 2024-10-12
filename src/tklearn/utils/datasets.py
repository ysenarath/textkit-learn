from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Union

import pandas as pd
from datasets import Dataset, DatasetDict
from tqdm import auto as tqdm

from tklearn.config import config as tk_config

T_BI = Dict[str, List[Any]]
T_BO = Union[Dict[str, List[Any]], List[Dict[str, Any]], pd.DataFrame]
T_I = Dict[str, Any]
T_O = Dict[str, Any]


class DatasetMapper:
    def __init__(
        self,
        func: Callable[[T_BI], T_BO],
        batched: bool = True,
        *,
        cache_dir: Path | str | None = None,
    ):
        if isinstance(cache_dir, str):
            cache_dir = Path(cache_dir)
        elif cache_dir is None:
            cache_dir = Path(tk_config.cache_dir) / "cache"
        self._cache_dir = cache_dir
        self.func = func
        self.batched = batched

    def __call__(self, *args, **kwargs) -> Any:
        return self.func(*args, **kwargs)

    @property
    def cache_dir(self) -> Path:
        if not self._cache_dir.exists():
            self._cache_dir.mkdir(parents=True)
        return self._cache_dir

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
        with tempfile.TemporaryDirectory(dir=self.cache_dir) as tmp_dir:
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


def map_dataset(
    dataset: Dataset | DatasetDict | T_BO,
    func: Callable[[T_BI], T_BO],
    batched: bool = True,
    batch_size: int = 8,
    func_kwargs: dict | None = None,
    verbose: bool = False,
    *,
    cache_dir: Path | str | None = None,
) -> Dataset:
    mapper = DatasetMapper(func, batched, cache_dir=cache_dir)
    return mapper.map(
        dataset,
        batch_size=batch_size,
        verbose=verbose,
        func_kwargs=func_kwargs,
    )


def example_2():
    def to_upper(x: T_BI) -> T_BO:
        return {"text": pd.Series(x["text"]).str.upper().tolist()}

    dataset = Dataset.from_dict({
        "text": ["hello world", "foo bar"],
        "label": [0, 1],
    })
    out = map_dataset(dataset, to_upper)
    print(type(out).__name__, out[:])
