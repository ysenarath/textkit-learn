from tklearn.datasets.stores.base import DataStore
from tklearn.datasets.stores.SQLAlchemyDataStore import SQLAlchemyDataStore

__all__ = [
    'DataStore',
    'SQLAlchemyDataStore',
]


def default_factory():
    return SQLAlchemyDataStore('default')
