import abc
from collections.abc import Sequence

__all__ = [
    'ObservableMixin',
    'ObserverMixin',
]


class ObserverList(Sequence):
    def __init__(self):
        self._data = []

    def __getitem__(self, item):
        return self._data[item]

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        yield from self._data

    def notify(self, *args, **kwargs):
        for item in self:
            item.notify(*args, **kwargs)

    def attach(self, observer):
        self._data.append(observer)


class ObservableMixin(object):
    def __init__(self):
        self.observers = ObserverList()


class ObserverMixin(object):
    @abc.abstractmethod
    def notify(self, *args, **kwargs):
        raise NotImplementedError
