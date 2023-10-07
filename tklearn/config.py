from __future__ import annotations
import os
from typing import Optional, Any
from collections import UserDict
import contextlib
from contextvars import ContextVar
from functools import wraps
from functools import partial as partial_func
import inspect

import yaml

__all__ = [
    "init_config",
]

load_yaml = partial_func(yaml.load, Loader=yaml.Loader)
dump_yaml = partial_func(yaml.dump, Dumper=yaml.Dumper)

__placeholder__ = object()


default = __placeholder__


def construct(cls, config=None, partial=False):
    if isinstance(cls, str):
        try:
            cls = load_yaml(f"!!python/name:{cls}")
        except:
            cls = load_yaml(f"!!python/name:__main__.{cls}")
    try:
        if config is None:
            config = {}
        if partial:
            return partial_func(cls, **config)
        return cls(**config)
    except Exception as ex:
        raise ex


class Config(UserDict):
    def __getitem__(self, key: str) -> Any:
        value = super().__getitem__(key)
        name = key.split("/")[-1]
        if name.startswith("@"):
            return construct(name[1:], value)
        if isinstance(value, dict):
            if "$type" in value:
                attrs = dict(value.items())
                cls = attrs.pop("$type")
                partial = attrs.pop("$partial", False)
                return construct(cls, config=Config(attrs).to_dict(), partial=partial)
            else:
                return Config(value)
        return value

    def __getattr__(self, name: str) -> Any:
        if name in self:
            return self[name]
        return None

    def to_dict(self):
        return {k: self[k] for k in self}


config_cv: ContextVar[Config] = ContextVar("config", default=Config())
scope_cv: ContextVar[Optional[str]] = ContextVar("scope", default=None)


class ConfigurableWrapper(object):
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
        self.params = Config(params)

    def set_params(self, **params) -> ConfigurableWrapper:
        return ConfigurableWrapper(self.builder, name=self.name, params=params)

    def _update_args(self, args, name):
        # func_name / class_name -> args
        if self.params:
            config = self.params
        else:
            config = config_cv.get()
            if name not in config:
                # function or class name not in config
                return args
            config = config[name]
        # update args
        for k, v in config.items():
            if k in args and args[k] is not __placeholder__:
                # already set - do not override
                continue
            args[k] = v
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


def configurable(builder):
    if not isinstance(builder, type):
        return wraps(builder)(ConfigurableWrapper(builder))
    wrapper = ConfigurableWrapper(builder)
    orig_init = builder.__init__

    @wraps(builder.__init__)
    def __init__(self, *args, **kwargs):
        orig_init(self, **wrapper.get_args(*args, **kwargs))

    builder.__init__ = __init__
    return builder


@contextlib.contextmanager
def config_scope(scope: str):
    current_scope = scope_cv.get()
    if current_scope:
        yield scope_cv.set(f"{current_scope}.{scope}")
    else:
        yield scope_cv.set(scope)
    scope_cv.set(current_scope)


def __getattr__(name):
    if name == "config":
        return config_cv.get()
    elif name == "scope":
        return scope_cv.get()
    raise AttributeError(name)


def init_config():
    config = config_cv.get()
    config.update(
        dict(
            resource_dir=os.path.expanduser("~/.tklearn"),
            logging=dict(
                level="WARNING",
                stream_handler=dict(
                    level="WARNING",
                    fmt="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                ),
            ),
        )
    )
