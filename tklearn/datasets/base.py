import os
from pathlib import Path
import typing
from typing import Any
import yaml

import intake

from tklearn.core import Environment
from tklearn.datasets.source import DataSource
from tklearn.datasets.stores import DataStore, default_factory
from tklearn.utils import logging

__all__ = [
    'Dataset',
]

logger = logging.get_logger(__name__)


class DataSourceCallback(object):
    def __init__(self, callback: typing.Callable = None, **kwargs) -> None:
        super(DataSourceCallback, self).__init__()
        self._callback = callback
        self._kwargs = kwargs

    def __getattr__(self, name: str) -> Any:
        if name.startswith('set_'):
            def _set_kwargs(value):
                self._kwargs[name[4:]] = value
                return self
            return _set_kwargs
        raise AttributeError(f'no attribute named {name}')

    def __call__(self, **kwargs):
        self._kwargs.update(kwargs)
        source = DataSource(**self._kwargs)
        return self._callback(source)


class Dataset(object):
    def __init__(
            self,
            stores: typing.Optional[list[DataStore]] = None,
            sources: typing.Optional[list[DataSource]] = None,
            env: Environment = None,
    ):
        super(Dataset, self).__init__()
        if env is None:
            env = Environment()
        self.__env__ = env
        self._stores: dict[str, DataStore] = {}
        self._sources: dict[str, DataSource] = {}
        default_found = False
        for item in stores or []:
            if item.name == 'default':
                default_found = True
            self.add_store(item)
        for item in sources or []:
            self.add_source(item)
        if env is None:
            env = Environment()
        if not default_found:
            self.add_store(default_factory())

    @property
    def env(self) -> Environment:
        return self.__env__

    @property
    def base_path(self) -> Path:
        return Path(self.env.config['DATASET_PATH'])

    @property
    def sources(self) -> tuple[DataSource]:
        return tuple(self._sources.values())

    @property
    def stores(self) -> tuple[DataStore]:
        return tuple(self._stores.values())

    def add_store(self, store: DataStore):
        if self._stores is None:
            self._stores = {}
        if store.name in self._stores:
            raise ValueError(f'store with name \'{store.name}\' already '
                             f'exists')
        setattr(store, '__env__', self.env)
        self._stores[store.name] = store
        return self

    def remove_store(self, name: str):
        del self._stores[name]
        return self

    def add_source(self, source: DataSource):
        if self._sources is None:
            self._sources = {}
        if source.name in self._sources:
            raise ValueError(f'source with name \'{source.name}\' already '
                             f'exists')
        setattr(source, '__env__', self.env)
        self._sources[source.name] = source
        return self

    def remove_source(self, name: str):
        del self._sources[name]
        return self

    def __getattr__(self, item):
        if item.startswith('add_source_'):
            source_type = item[11:]  # len('add_source_') = 11
            source = getattr(intake, f'open_{source_type}')
            driver: str = source.__module__ + '.' + source.__qualname__
            return DataSourceCallback(
                self.add_source, driver=driver
            )
        raise AttributeError(f'no attribute named {item}')

    def dict(self):
        return {
            'stores': [store.dict() for store in self.stores],
            'sources': [source.dict() for source in self.sources],
        }

    @classmethod
    def load(self, path: str, create: bool = True, env: Environment = None)\
            -> 'Dataset':
        path = str(path)
        if env is None:
            env = Environment()
        # create the dataset path if it does not exist
        base_path = env.format(path)
        env.config['DATASET_PATH'] = base_path
        if isinstance(base_path, str):
            base_path = Path(base_path)
        if not base_path.exists():
            if create:
                os.makedirs(base_path)
            else:
                raise ValueError(f'path \'{base_path}\' not found')
        dataset_yaml_path = base_path / 'dataset.yaml'
        if not dataset_yaml_path.exists():
            if create:
                dataset_yaml_path.touch()
            else:
                raise ValueError(f'dataset.yaml not found in \'{base_path}\'')
        # read the dataset.yaml file and create dataset
        dataset_text = dataset_yaml_path.read_text()
        dataset_yaml = yaml.safe_load(dataset_text)
        if dataset_yaml is None:
            dataset_yaml = {}
        stores_ = []
        if 'stores' in dataset_yaml:
            for store in dataset_yaml['stores']:
                stores_.append(DataStore(**store))
        else:
            stores_.append(default_factory())
        sources = []
        for source in dataset_yaml.get('sources', []):
            sources.append(DataSource(**source))
        return Dataset(stores=stores_, sources=sources, env=env)

    def sync(self, verbose=False):
        """Sync the data from sources to stores.

        Notes
        -----
        This method will delete all data (that has the same name as the
        source) from the stores and write the data from the sources to the
        stores.

        Parameters
        ----------
        verbose: bool
            Whether to print verbose output.

        Returns
        -------
        None
            None
        """
        i = 1
        total = len(self.sources) * len(self.stores)
        for source in self.sources:
            for store in self.stores:
                if verbose:
                    desc = f'syncing {source.name} to {store.name} ({i}/{total})'
                    logger.info(desc)
                store.write(source, verbose=verbose)
                i += 1
        return self

    def dump(self, path: str, verbose=False):
        """Save the dataset to a yaml file.

        Parameters
        ----------
        path: str
            The path to dump the dataset to.
        verbose: bool
            Whether to print verbose output.

        Returns
        -------
        None
            None
        """
        if path is None:
            raise ValueError('path cannot be None')
        # convert to Path object after formatting
        path = Path(self.env.format(str(path)))
        if not path.exists():
            # create the path if it does not exist
            os.makedirs(path)
        dataset_yaml = yaml.dump(self.dict())
        dataset_yaml_path = path / 'dataset.yaml'
        dataset_yaml_path.write_text(dataset_yaml)
        copy = Dataset.load(path)
        for key, target in copy._stores.items():
            source = self._stores[key]
            target.write(source, verbose=verbose)
        return copy

    def __iter__(self):
        for store in self.stores:
            yield from store
