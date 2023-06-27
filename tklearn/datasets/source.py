import typing

from intake.source.base import DataSource as DataSource_
from intake.catalog import Catalog
from intake.catalog.local import LocalCatalogEntry


__all__ = [
    'DataSource',
]


class DataSource(object):
    def __init__(
            self,
            name: str,
            driver: str,
            description: str = '',
            args: typing.Dict[str, typing.Any] = {},
            metadata: typing.Dict[str, typing.Any] = {},
    ) -> None:
        super(DataSource, self).__init__()
        self.name = name
        self.description = description
        self.driver = driver
        self.args = args
        self.metadata = metadata
        self._source: typing.Optional[DataSource_] = None
        self.__env__ = None

    @property
    def env(self):
        return self.__env__

    def _get_source(self) -> DataSource_:
        if not hasattr(self, '_source') or self._source is None:
            kwargs = self.dict(exclude={'store'})
            args = kwargs.pop('args', {})
            if args is None:
                args = {}
            args['urlpath'] = self.env.format(args['urlpath'])
            kwargs['args'] = args
            cat = Catalog.from_dict({
                self.name: LocalCatalogEntry(**kwargs)
            })
            self._source = getattr(cat, self.name)
        return self._source

    def _df_to_dict(self, df, **kwargs) -> typing.Any:
        container = self._get_source().container
        if container != 'dataframe':
            raise ValueError('data source is not a dataframe')
        return df.to_dict(orient='records')

    def read(self):
        ds = self._get_source()
        df = ds.read()
        return self._df_to_dict(df)

    def read_chunked(self):
        offset = 0
        ds = self._get_source()
        for df in ds.read_chunked():
            yield offset, self._df_to_dict(df)
            offset += len(df)

    def dict(self, exclude: typing.Set[str] = None):
        if exclude is None:
            exclude = set()
        kwargs = {
            k: v for k, v in self.__dict__.items()
            if k not in exclude and not k.startswith('_')
        }
        return kwargs

    def __len__(self) -> int:
        return len(self._get_source().to_dask())

    def __iter__(self) -> iter:
        for offset, chunk in self.read_chunked():
            for doc in chunk:
                yield doc
