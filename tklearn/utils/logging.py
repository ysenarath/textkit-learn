import logging

from tqdm import auto as tqdm

from tklearn.config import config

__all__ = [
    "get_logger",
    "ProgressBar",
    "Progbar",
]


class ProgressBar(tqdm.tqdm):
    pass


Progbar = ProgressBar  # alias


def get_logger(name: str) -> logging.Logger:
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
    if not config.logging:
        logger.setLevel(logging.DEBUG)
        logger.debug("logger configuration not found, using 'ERROR'")
        logger.setLevel(logging.ERROR)
        return logger
    logger_level = config.logging.level
    if not logger_level:
        logger.debug("logger level configuration not found, using 'ERROR'")
        logger.setLevel(logging.ERROR)
    logger.setLevel(logger_level)
    if not config.logging.stream_handler:
        logger.debug("stream handler configuration not found, using default")
        return logger
    handler = logging.StreamHandler()
    handler.setLevel(config.logging.stream_handler.level or logger_level)
    fmt = "[%(asctime)s] %(levelname)s %(name)s: %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(
        config.logging.stream_handler.fmt or fmt,
        datefmt=config.logging.stream_handler.datefmt or datefmt,
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
