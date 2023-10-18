from typing import Optional, Dict, Any
import logging

from tqdm import auto as tqdm

from tklearn.config import configurable, config_scope

__all__ = [
    "get_logger",
    "ProgressBar",
    "Progbar",
]


class ProgressBar(tqdm.tqdm):
    pass


Progbar = ProgressBar  # alias


def default_stream_handler(
    level: Optional[str],
    fmt: str = "[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt: str = "%Y-%m-%d %H:%M:%S",
) -> logging.StreamHandler:
    handler = logging.StreamHandler()
    handler.setLevel(level)
    formatter = logging.Formatter(fmt, datefmt=datefmt)
    handler.setFormatter(formatter)
    return handler


@configurable("logging")
def get_logger(
    name: str,
    level: Optional[str] = None,
    stream_handler: Optional[Dict[str, Any]] = None,
) -> logging.Logger:
    """Get default logger by name.

    Parameters
    ----------
    name : str
        Name of the logger.

    Returns
    -------
    logging.Logger
        Logger object.
    """
    logger = logging.getLogger(name)
    if level is None:
        logger.debug("logger configuration not found, using 'ERROR'")
        level = logging.ERROR
    logger.setLevel(level)
    if stream_handler is not None:
        handler = default_stream_handler(**stream_handler)
        logger.addHandler(handler)
    return logger
