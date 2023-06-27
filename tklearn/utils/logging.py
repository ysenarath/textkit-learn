import logging

__all__ = [
    'get_logger',
]


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
    if name == 'sqlalchemy':
        return logger
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
