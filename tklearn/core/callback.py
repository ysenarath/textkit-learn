from functools import partial
import typing
from tklearn.core.observer import Observer, Observable

__all__ = [
    'Callback',
    'CallbackFunction',
    'callback',
]


class CallbackFunction(Observer):
    def __init__(self, func, name=None, obj=None) -> None:
        self._name = name or func.__name__
        self._func = func
        self._obj = obj

    @property
    def name(self):
        return self._name

    def __call__(self, *args, **kwargs):
        if self._obj is not None:
            return self._func(self._obj, *args, **kwargs)
        return self._func(*args, **kwargs)

    def update(self, event, *args, **kwargs) -> None:
        if event != self.name:
            return
        self(*args, **kwargs)


class CallbackDescriptor:
    def __init__(self, name, func) -> None:
        self._name = name
        self._func = func

    def __get__(self, obj, objtype=None):
        if not issubclass(objtype, Callback):
            raise AttributeError(
                'callbacks functions can only be used within a Callback object'
            )
        if obj is None:
            return self
        return CallbackFunction(self._func, name=self._name, obj=obj)

    def __set__(self, obj, value):
        raise AttributeError('Cannot set attribute')

    def __delete__(self, obj):
        raise AttributeError('Cannot delete attribute')


class Callback(Observable, Observer):
    def __init__(self, callbacks=None) -> None:
        super(Callback, self).__init__()
        for callback in callbacks or []:
            self.attach(callback)

    @property
    def _callbacks(self):
        return self._observers

    def _on_callback(self, name):
        def _callback(*args, **kwargs):
            self.notify(name, *args, **kwargs)
        return _callback

    def __getattr__(self, name):
        if name.startswith('on_'):
            name = name[3:]
            return self._on_callback(name)
        raise AttributeError(
            f'{self.__class__.__name__} object has no attribute {name}'
        )

    def update(self, *args, **kwargs) -> None:
        self.notify(*args, **kwargs)

    def attach(
        self,
        callback: typing.Union['CallbackFunction', 'Callback', type]
    ) -> None:
        if isinstance(callback, type):
            callback = callback()
        if isinstance(callback, Callback):
            for attr in dir(callback):
                cf = getattr(callback, str(attr))
                if not isinstance(cf, CallbackFunction):
                    continue
                self.attach(cf)
        elif isinstance(callback, CallbackFunction):
            super(Callback, self).attach(callback)
        else:
            raise TypeError(
                'callback must be an instance of Callback or CallbackFunction'
            )


def callback(func, name=None) -> typing.Union[CallbackDescriptor, partial]:
    if isinstance(func, str):
        return partial(callback(name=func))
    name = name or func.__name__
    if name.startswith('on_'):
        name = name[3:]
    desc = CallbackDescriptor(name, func)
    desc.__name__ = func.__name__
    return desc
