from __future__ import annotations

import functools
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Union

import pandas as pd
from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict
from datasets import load_dataset as hf_load_dataset
from tqdm import auto as tqdm

from tklearn.config import config
from tklearn.core.document import Document

T_BI = Dict[str, List[Any]]
T_BO = Union[Dict[str, List[Any]], List[Dict[str, Any]], pd.DataFrame]
T_I = Dict[str, Any]
T_O = Dict[str, Any]


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
) -> Generator[Document, None, None]:
    split = kwargs.get("split", None)
    if split is None:
        n = dataset.num_rows
    else:
        n = dataset[split].num_rows
    for i in range(*args):
        if i >= n:
            break
        j = n + i if i < 0 else i
        yield Document.from_dataset(dataset, j, split)


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
