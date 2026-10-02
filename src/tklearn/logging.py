from __future__ import annotations

import functools
import logging
from typing import Callable, Optional, TypeVar, overload

from tklearn import config

__all__ = [
    "get_logger",
    "log_on_exception",
]

_LOGGING_TEMPLATE = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

T = TypeVar("T")

CRITICAL = logging.CRITICAL
FATAL = CRITICAL
ERROR = logging.ERROR
WARNING = logging.WARNING
WARN = WARNING
INFO = logging.INFO
DEBUG = logging.DEBUG
NOTSET = logging.NOTSET


def get_logger(
    name: str | None = None,
    level: int | str | None = None,
    fmt: str = _LOGGING_TEMPLATE,
) -> logging.Logger:
    """Return a logger with a console handler.

    Parameters
    ----------
    name : str, optional
        Logger name, usually ``__name__``.
    level : int or str, optional
        Log level; DEBUG when ``config.debug`` is set, otherwise INFO.
    fmt : str
        Format of console messages.
    """
    if level is None:
        level = DEBUG if config.debug else INFO
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # add the console handler once, however often the logger is requested
    if not any(getattr(h, "_tklearn", False) for h in logger.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(fmt))
        handler._tklearn = True
        logger.addHandler(handler)
    return logger


@overload
def log_on_exception(logger: logging.Logger) -> Callable[[T], T]: ...


@overload
def log_on_exception(
    func: T, logger: Optional[logging.Logger] = None
) -> T: ...


def log_on_exception(
    func_or_logger: T | logging.Logger, logger: Optional[logging.Logger] = None
) -> T | Callable[..., T]:
    """Log exceptions raised by a function before re-raising them."""
    if isinstance(func_or_logger, logging.Logger):
        return functools.partial(log_on_exception, logger=func_or_logger)

    func = func_or_logger

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            (logger or get_logger(func.__module__)).error(
                f"Error in {func.__name__}: {e}"
            )
            raise

    return wrapper
