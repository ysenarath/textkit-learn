import functools
from pathlib import Path
import shutil
import tempfile
import typing

from joblib import Memory

from tklearn.config import config
from tklearn.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "mkdtemp",
    "cache",
]


def getcachedir() -> Path:
    return Path(config.resource_dir) / "cache"


def clear(prefix: typing.Optional[str] = None, create: bool = True):
    # remove cache dir
    cachedir = getcachedir()
    if prefix is None and cachedir.exists():
        shutil.rmtree(cachedir)
    else:
        for path in cachedir.glob(f"{prefix}*"):
            shutil.rmtree(path)
    # create cache dir
    if create:
        cachedir.mkdir(parents=True, exist_ok=True)


def mkdtemp(
    cachedir: typing.Union[str, Path, None] = None,
    prefix: typing.Optional[str] = None,
    suffix: typing.Optional[str] = None,
) -> Path:
    if cachedir is None:
        resource_dir = Path(config.resource_dir)
        try:
            resource_dir.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            logger.warning(
                f"resource directory '{resource_dir}' is not writable, "
                "using temporary directory instead."
            )
            resource_dir = None
        else:
            # set cache dir
            cachedir = getcachedir()
    if cachedir is not None:
        cachedir = Path(cachedir)
        cachedir.mkdir(parents=True, exist_ok=True)
    path = tempfile.mkdtemp(
        prefix=prefix,
        suffix=suffix,
        dir=str(cachedir),
    )
    return Path(path)


default_memory = Memory(getcachedir() / "joblib", verbose=0)


def cache(
    func=None,
    *,
    ignore=None,
    verbose=None,
    mmap_mode=False,
    cache_validation_callback=None,
    memory=default_memory,
):
    if func is None:
        return functools.partial(
            cache,
            ignore=ignore,
            verbose=verbose,
            mmap_mode=mmap_mode,
            cache_validation_callback=cache_validation_callback,
        )
    return memory.cache(
        func=func,
        ignore=ignore,
        verbose=verbose,
        mmap_mode=mmap_mode,
        cache_validation_callback=cache_validation_callback,
    )


def lru_cache(func: typing.Callable):
    @functools.lru_cache
    @functools.wraps(func)
    def decorator(*args, **kwargs):
        return cache(func)(*args, **kwargs)

    return decorator
