from __future__ import annotations

from typing import Any, Callable, Concatenate, Generic, Self

from typing_extensions import ParamSpec, TypeVar

__all__ = [
    "MethodMixin",
    "method",
]

P = ParamSpec("P")
T = TypeVar("T")


class Method(Generic[P, T]):
    def __init__(self, container: Any, default: Callable[P, T]) -> None:
        self.container = (
            container  # the instance of the class that this method is bound to
        )
        self.func = default

    def register(self, func: Any) -> Any:
        self.func = func
        return func

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T:
        return self.func(self.container, *args, **kwargs)


class MethodDescriptor(Generic[P, T]):
    def __init__(self, func: Callable[Concatenate[Any, P], T]) -> None:
        super().__init__()
        self.func = func

    def __set_name__(self, owner: Any, name: str) -> None:
        self.name = name
        self.private_name = f"_{name}"

    def __get__(self, instance: Any, owner: Any) -> Method[P, T]:
        if not hasattr(instance, self.private_name):
            method = Method(instance, default=self.func)
            setattr(instance, self.private_name, method)
        return getattr(instance, self.private_name)


def method(func: Callable[Concatenate[Any, P], T]) -> MethodDescriptor[P, T]:
    return MethodDescriptor(func)


class MethodMixin:
    def register(self, name: str, func: Callable) -> Self:
        method = getattr(self, name)
        if not isinstance(method, Method):
            msg = f"'{name}' is not a registerable method"
            raise TypeError(msg)
        method.register(func)
        return self
