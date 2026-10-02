from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from typing import Any, Generic, TypeVar, overload

__all__ = [
    "Registry",
]

T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])


class Registry(Generic[T]):
    """Map names to classes or factory functions.

    Registries are for choosing an implementation by name, for example from a
    configuration file. Everything a registry builds can also be constructed
    directly.

    Parameters
    ----------
    kind : str
        What the registry holds (e.g. ``"embedding"``), used in errors.

    Examples
    --------
    >>> BACKBONES = Registry("backbone")
    >>> @BACKBONES.register("transformer")
    ... class TransformerBackbone(Backbone): ...
    >>> BACKBONES.create("transformer", model_name_or_path="bert-base-uncased")
    >>> BACKBONES.from_config({"type": "transformer", "model_name_or_path": "..."})
    """

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._entries: dict[str, Callable[..., T]] = {}

    @overload
    def register(self, name: str) -> Callable[[F], F]: ...
    @overload
    def register(self, name: str, factory: F) -> F: ...
    def register(self, name: str, factory: F | None = None) -> Any:
        """Register `factory` under `name`; usable as a decorator."""

        def decorator(factory: F) -> F:
            if name in self._entries and self._entries[name] is not factory:
                msg = f"{self.kind} {name!r} is already registered"
                raise ValueError(msg)
            self._entries[name] = factory
            return factory

        if factory is None:
            return decorator
        return decorator(factory)

    def get(self, name: str) -> Callable[..., T]:
        """Return the class or factory registered under `name`."""
        try:
            return self._entries[name]
        except KeyError:
            available = ", ".join(sorted(self._entries)) or "none"
            msg = f"unknown {self.kind} {name!r}; available: {available}"
            raise KeyError(msg) from None

    def create(self, name: str, /, *args: Any, **kwargs: Any) -> T:
        """Construct the entry registered under `name`."""
        return self.get(name)(*args, **kwargs)

    def from_config(self, config: Mapping[str, Any], key: str = "type") -> T:
        """Construct from a mapping whose `key` names the entry.

        The remaining items are passed as keyword arguments.
        """
        kwargs = dict(config)
        try:
            name = kwargs.pop(key)
        except KeyError:
            msg = f"{self.kind} config is missing the {key!r} key"
            raise KeyError(msg) from None
        return self.create(name, **kwargs)

    def names(self) -> list[str]:
        return sorted(self._entries)

    def __contains__(self, name: object) -> bool:
        return name in self._entries

    def __iter__(self) -> Iterator[str]:
        return iter(self.names())

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.kind!r}, {self.names()})"
