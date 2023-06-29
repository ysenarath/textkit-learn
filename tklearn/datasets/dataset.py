from itertools import tee
import json
import os
import tempfile
import typing
from collections.abc import Mapping, Sequence
from pathlib import Path
import weakref

import pyarrow as pa
import pyarrow.dataset as ds
import pandas as pd
from tqdm import auto as tqdm

from tklearn.datasets.loader import get_dataset_loader
from tklearn.core.version import Version, versions

__all__ = [
    'Dataset',
    'load_dataset',
    'write_dataset',
]

InputType = typing.Union[
    pa.Table,
    pd.DataFrame,
    Mapping,
    Sequence,
]


class BaseDataset(object):
    def __init__(self, base_path):
        if isinstance(base_path, str):
            base_path = Path(base_path)
        self.base_path: Path = base_path
        self._pa_dataset: ds.Dataset = ds.dataset(
            base_path, format='parquet'
        )

    def _replace_schema_metadata(self, **metadata):
        metadata = {
            key: json.dumps(value) for key, value in metadata.items()
        }
        updated_schema = self._pa_dataset.schema.with_metadata(metadata)
        self._pa_dataset = self._pa_dataset.replace_schema(updated_schema)

    @property
    def version(self) -> Version:
        version = None
        if self._pa_dataset.schema.metadata is not None:
            if b'version' in self._pa_dataset.schema.metadata:
                version = json.loads(
                    self._pa_dataset.schema.metadata[b'version']
                )
                version = Version(**version)
        if version is None:
            return versions.default
        return version

    @version.setter
    def version(self, version: typing.Union[Version, str]):
        if isinstance(version, str):
            version = Version(version)
        self._replace_schema_metadata(
            version=version.dict(),
            metadata=self.metadata,
        )

    @property
    def metadata(self) -> Mapping:
        metadata = None
        if self._pa_dataset.schema.metadata is not None:
            if b'metadata' in self._pa_dataset.schema.metadata:
                metadata = self._pa_dataset.schema.metadata[b'metadata']
                metadata = json.loads(metadata)
        if metadata is None:
            return {}
        return metadata

    @metadata.setter
    def metadata(self, metadata: Mapping[str, typing.Any]):
        self._replace_schema_metadata(
            version=self.version.dict(),
            metadata=metadata,
        )

    def __getitem__(self, item):
        if isinstance(item, int):
            return self._pa_dataset.take([item]).to_pylist()[0]
        raise TypeError(f'unsupported index type: {type(item).__name__}')

    def __len__(self):
        return self._pa_dataset.count_rows()

    def __iter__(self):
        for batch in self._pa_dataset.to_batches():
            for doc in batch.to_pylist():
                yield doc

    def _iter_with_pbar(self, iterable, pbar=None):
        for item in iterable:
            yield item
            if pbar is not None:
                pbar.update(1)

    def map(self, func: typing.Callable, verbose: bool = False):
        if not callable(func):
            raise TypeError(f'func must be callable, got {type(func)}')
        batches = self._pa_dataset.to_batches()
        pbar = None
        if verbose:
            pbar = tqdm.tqdm(
                total=len(self),
                leave=False,
                ncols=100,
                # colour='green',
            )
        mapped_batches_iter = (pa.RecordBatch.from_pylist(list(
            map(func, self._iter_with_pbar(batch.to_pylist(), pbar))
        )) for batch in batches)
        mapped_batches_iter, first = tee(mapped_batches_iter)
        _mapped_table = ds.Scanner.from_batches(
            mapped_batches_iter,
            schema=next(first).schema,
        )
        return Dataset(
            _mapped_table,
            metadata=self.metadata,
            version=self.version,
        )


class Dataset(BaseDataset):
    def __init__(
            self,
            dataset: typing.Union[pa.Table, ds.Dataset, ds.Scanner],
            metadata: typing.Optional[Mapping[str, typing.Any]] = None,
            version: typing.Union[Version, str, None] = None,
            base_path: typing.Optional[typing.Union[str, Path]] = None,
            **kwargs
    ):
        dataset = self._get_dataset(dataset)
        if metadata is None:
            metadata = {}
        if version is None:
            version = versions.default
        self._tempdir = None
        if base_path is None:
            self._tempdir = tempfile.TemporaryDirectory()
            base_path = Path(self._tempdir.name)
        if isinstance(base_path, str):
            base_path = Path(base_path)
        ds.write_dataset(
            dataset,
            base_dir=base_path,
            format='parquet',
            **kwargs,
        )
        # delete reference to the table
        del dataset
        self._finalizer = weakref.finalize(self, self._cleanup)
        super(Dataset, self).__init__(base_path=base_path)
        self.metadata = metadata
        self.version = version

    def _cleanup(self):
        if self._tempdir is None:
            return
        if self._finalizer.detach() or os.path.exists(self.name):
            self._tempdir.cleanup()

    def close(self):
        self._cleanup()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _get_dataset(self, table, **kwargs):
        factory_method = None
        if isinstance(table, pd.DataFrame):
            factory_method = pa.Table.from_pandas
        elif isinstance(table, Mapping):
            factory_method = pa.Table.from_pydict
        elif isinstance(table, Sequence):
            factory_method = pa.Table.from_pylist
        elif not isinstance(table, (pa.Table, ds.Dataset, ds.Scanner)):
            raise TypeError(f'unsupported table type: {type(table)}')
        if factory_method:
            table = factory_method(table, **kwargs)
        return table

    @classmethod
    def load_from_disk(cls, path, **kwargs):
        self = super(Dataset, cls).__new__(cls)
        super(Dataset, self).__init__(base_path=path)
        return self

    def save_to_disk(self, path: typing.Union[str, Path], **kwargs):
        if isinstance(path, str):
            path = Path(path)
        ds.write_dataset(
            self._pa_dataset,
            base_dir=path / str(self.version),
            format='parquet',
            **kwargs,
        )
        # load the saved dataset from fs
        return self.load_from_disk(path=path)


def write_dataset(
        dataset: Dataset,
        path: typing.Optional[typing.Union[str, Path]] = None,
        **kwargs
):
    if isinstance(path, str):
        path = Path(path)
    return Dataset.save_to_disk(dataset, path=path, **kwargs)


def load_dataset(name_or_path, version=versions.latest, **kwargs) -> Dataset:
    base_path = Path(name_or_path)
    version = None
    if (
        base_path.exists()
        and base_path.is_dir()
        and (
            version == versions.latest
            or version is not None
        )
    ):
        avail_versions = [
            Version(version)
            for version in base_path.listdir()
        ]
        if len(avail_versions) != 0:
            version = max(avail_versions)
    if version is not None:
        try:
            return Dataset.load_from_disk(
                path=base_path / str(version),
                **kwargs,
            )
        except FileNotFoundError:
            pass
    # load dataset with the name
    try:
        loader = get_dataset_loader(name_or_path)
    except KeyError as e:
        raise ValueError(f'the dataset {name_or_path} is not supported') from e
    it = []
    for split in loader.splits:
        it += loader(split=split, binarize=False, return_dict=True)
    if version is None:
        version = versions.default
    metadata = {'builder_name': loader.name}
    if loader.citation:
        metadata['citation'] = loader.citation
    if loader.description:
        metadata['description'] = loader.description
    return Dataset(it, version=version, metadata=metadata, **kwargs)
