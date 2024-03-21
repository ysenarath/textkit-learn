import logging
from typing import Optional, Sequence, Union

__all__ = [
    "get_logger",
]


def get_logger(
    name: Optional[str] = None,
    level: Union[int, str, None] = None,
    handlers: Optional[Sequence[str]] = None,
    formatter: Optional[str] = None,
) -> logging.Logger:
    """
    Get a logger with the given name and level.

    Parameters
    ----------
    name : str, optional
        Name of the logger.
    level : int or str, optional
        Logging level.
    formatter : str, optional
        Formatter string.
    handlers : Sequence[str], optional
        Sequence of handlers to use.

    Returns
    -------
    logging.Logger
        Logger instance.
    """
    logger = logging.getLogger(name)
    if level is not None:
        logger.setLevel(level)
    if handlers is None:
        handlers = ()
    elif isinstance(handlers, str):
        handlers = (handlers,)
    if "console" in handlers:
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.DEBUG)
        if not isinstance(formatter, logging.Formatter):
            # Create a formatter
            formatter = logging.Formatter(formatter)
        # add it to the handlers
        stream_handler.setFormatter(formatter)
        # Add the handlers to the logger
        logger.addHandler(stream_handler)
    return logger
