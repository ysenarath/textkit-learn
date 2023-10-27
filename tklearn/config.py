from __future__ import annotations
import os
from typing import Iterator, Optional, Any
from collections.abc import Mapping
import contextlib
from contextvars import ContextVar
from functools import wraps
from functools import partial as partial_func
import inspect

from werkzeug.local import LocalProxy
import yaml

__all__ = [
    "init_config",
    "config_scope",
]

load_yaml = partial_func(yaml.load, Loader=yaml.Loader)
dump_yaml = partial_func(yaml.dump, Dumper=yaml.Dumper)

# config sentinel

__placeholder__ = object()


default = __placeholder__

# config utils


def construct(cls, config=None, partial=False):
    if isinstance(cls, str):
        try:
            cls = load_yaml(f"!!python/name:{cls}")
        except:
            cls = load_yaml(f"!!python/name:__main__.{cls}")
    try:
        if config is None:
            config = {}
        if not isinstance(config, dict):
            raise TypeError(f"config must be a dict, found {type(config).__name__}")
        if partial:
            return partial_func(cls, **config)
        return cls(**config)
    except Exception as ex:
        raise ex


# config main


class FrozenConfig(Mapping):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.data = dict(*args, **kwargs)

    def __iter__(self) -> Iterator:
        yield from self.data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, key: str) -> Any:
        value = self.data[key]
        name = key.split("/")[-1]  # remove the scope from the key
        # if name.startswith("@"):
        #     return construct(name[1:], value)
        if isinstance(value, dict):
            if "$type" in value:
                attrs = dict(value.items())
                cls = attrs.pop("$type")
                partial = attrs.pop("$partial", False)
                return construct(
                    cls,
                    config=FrozenConfig(attrs).to_dict(),
                    partial=partial,
                )
            else:
                return FrozenConfig(value)
        return value

    def __getattr__(self, name: str) -> Any:
        if name in self:
            return self[name]
        return None

    def to_dict(self) -> dict:
        out = {}
        for k, v in self.items():
            out[k] = v.to_dict() if isinstance(v, FrozenConfig) else v
        return out

    def __reduce__(self) -> tuple:
        return (self.__class__, (self.data,))


class ContextVarConfig(FrozenConfig):
    def update(self, items):
        config: FrozenConfig = config_cv.get()
        config.data.update(items)
        config_cv.set(config)

    def __setitem__(self, key, value):
        self.update({key: value})

    def clear(self):
        config_cv.reset()


config_cv: ContextVar[ContextVarConfig] = ContextVar(
    "config", default=ContextVarConfig()
)

config: ContextVarConfig = LocalProxy(config_cv)

# config scoping

scope_cv: ContextVar[Optional[str]] = ContextVar("scope", default=None)


@contextlib.contextmanager
def config_scope(scope: str):
    current_scope = scope_cv.get()
    if current_scope:
        yield scope_cv.set(f"{current_scope}.{scope}")
    else:
        yield scope_cv.set(scope)
    scope_cv.set(current_scope)


# config wrappers (auto init)


class ConfigWrapper(object):
    """Configure the object."""

    def __init__(self, builder, name=None, params=None):
        self.builder = builder
        self.name = (
            name
            if name is not None
            else builder.__qualname__
            if hasattr(builder, "__qualname__")
            else builder.__name__
            if hasattr(builder, "__name__")
            else None
        )
        if self.name is None:
            raise ValueError("name is required")
        if params is None:
            params = {}
        self.params = FrozenConfig(**params)

    def set_params(self, **params) -> ConfigWrapper:
        return ConfigWrapper(self.builder, name=self.name, params=params)

    def _update_args(self, args, name):
        # func_name / class_name -> args
        if self.params:
            config = self.params
        else:
            config = config_cv.get()
            if name not in config:
                # function or class name or alias is not in config
                return args
            config = config[name]
        # update args
        for k in config.keys():
            if k in args and args[k] is not __placeholder__:
                # already set - do not override
                continue
            args[k] = config[k]
        return args

    def get_args(self, *args, **kwargs):
        args = inspect.signature(self.builder).bind_partial(*args, **kwargs)
        args = args.arguments
        args = self._update_args(args, self.name)
        scope = scope_cv.get()
        if scope is not None:
            args = self._update_args(
                args,
                f"{scope}/{self.name}",
            )
        args = inspect.signature(self.builder).bind_partial(**args)
        args.apply_defaults()
        args = args.arguments
        missing = []
        for k, v in args.items():
            if v is __placeholder__:
                missing.append(k)
        if missing:
            raise ValueError(f"missing {len(missing)} required arguments: {missing}")
        return args

    def __call__(self, *args, **kwargs):
        return self.builder(**self.get_args(*args, **kwargs))


def configurable(builder=None, **kwargs):
    if isinstance(builder, str):
        return partial_func(
            configurable,
            name=builder,
        )
    elif builder is None:
        return partial_func(configurable, **kwargs)
    name = kwargs.get("name", None)
    wrapper = ConfigWrapper(builder, name=name)
    if not isinstance(builder, type):  # if not a class

        @wraps(builder)
        def builder_wrapper(*args, **kwargs):
            return wrapper(*args, **kwargs)

        return builder_wrapper

    orig_init = builder.__init__

    @wraps(builder.__init__)
    def __init__(self, *args, **kwargs):
        orig_init(self, **wrapper.get_args(*args, **kwargs))

    builder.__init__ = __init__
    return builder


# tklearn default configs


def init_config():
    global config
    config.update(
        dict(
            resource=dict(
                path=os.path.expanduser("~/.tklearn"),
            ),
            logging=dict(
                level="DEBUG",
                stream_handler=dict(
                    level="DEBUG",
                    fmt="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                ),
            ),
        )
    )
