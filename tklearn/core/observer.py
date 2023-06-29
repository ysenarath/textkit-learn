__all__ = [
    'Observer',
    'Observable',
]


class Observer(object):
    def update(self, *args, **kwargs) -> None:
        raise NotImplementedError


class Observable(object):
    def __init__(self, *args, **kwargs) -> None:
        self._observers = []
        super(Observable, self).__init__(*args, **kwargs)

    def attach(self, observer: Observer) -> None:
        if not isinstance(observer, Observer):
            raise TypeError(
                'observer must be an instance of Observer'
            )
        self._observers.append(observer)

    def detach(self, observer: Observer) -> None:
        self._observers.remove(observer)

    def notify(self, *args, **kwargs) -> None:
        for observer in self._observers:
            observer.update(*args, **kwargs)
