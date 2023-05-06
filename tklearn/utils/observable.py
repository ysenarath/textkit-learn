"""A module for making objects observable.

An observable object is an object that can notify observers of changes.
An observer is an object that can be notified of changes to an observable object.
"""
import abc
from collections.abc import Sequence

__all__ = [
    'ObservableMixin',
    'ObserverMixin',
]


class ObserverList(Sequence):
    """A list of observers that can be notified of changes to an observable object."""

    def __init__(self):
        self._data = []

    def __getitem__(self, item):
        """Get the observer at the given index.

        Parameters
        ----------
        item : int
            The index of the observer to get.

        Returns
        -------
        ObserverMixin
            The observer at the given index.
        """
        return self._data[item]

    def __len__(self):
        """Get the number of observers.
        
        Returns
        -------
        int
            The number of observers.
        """
        return len(self._data)

    def __iter__(self):
        """Get an iterator over the observers.
        
        Returns
        -------
        Iterator[ObserverMixin]
            An iterator over the observers.
        """
        yield from self._data

    def notify(self, *args, **kwargs):
        """Notify all observers of a change.
        
        Parameters
        ----------
        args
            The positional arguments to pass to the observers.
        kwargs
            The keyword arguments to pass to the observers.
        """
        for item in self:
            item.notify(*args, **kwargs)

    def attach(self, observer):
        """Attach an observer to the list.
        
        Parameters
        ----------
        observer : ObserverMixin
            The observer to attach.

        Raises
        ------
        TypeError
            If the observer is not an instance of ObserverMixin.
        """
        if not isinstance(observer, ObserverMixin):
            raise TypeError('observer must be an instance of \'ObserverMixin\'')
        self._data.append(observer)


class ObservableMixin(object):
    """A mixin class for making an object observable.
    
    An observable object is an object that can notify observers of changes.
    
    Attributes
    ----------
    observers : ObserverList
        The list of observers.
    """

    def __init__(self):
        """A mixin class for making an object observable."""
        self.observers = ObserverList()


class ObserverMixin(object):
    """An abstract mixin class for making an object an observer.
    
    An observer is an object that can be notified of changes to an observable object.

    Methods
    -------
    notify(*args, **kwargs)
        Notify the observer of a change.
    """

    @abc.abstractmethod
    def notify(self, *args, **kwargs):
        """Notify the observer of a change.
        
        Parameters
        ----------
        args
            The positional arguments to pass to the observer.

        kwargs
            The keyword arguments to pass to the observer.
        """
        raise NotImplementedError
