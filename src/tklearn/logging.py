from __future__ import annotations

import logging
from dataclasses import field
from typing import Union, cast

from nightjar import BaseConfig

__all__ = [
    "get_logger",
]

_LOGGING_TEMPLATE = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


class LoggingFormatterConfig(BaseConfig):
    fmt: str | None = _LOGGING_TEMPLATE
    datefmt: Union[str, None] = None
    style: str = "%"
    validate: bool = True


class LoggingConfig(BaseConfig):
    level: str = "INFO"
    fmt: LoggingFormatterConfig = field(default_factory=LoggingFormatterConfig)

    def __post_init__(self):
        self.level = self.level.upper()


class LoggerWithConfig(logging.Logger):
    config: LoggingConfig


def get_logger(
    name: str | None = None, config: LoggingConfig | None = None
) -> LoggerWithConfig:
    if config is None:
        config = LoggingConfig()
    logger = cast(LoggerWithConfig, logging.getLogger(name))
    logger.config = config
    logger.setLevel(config.level)
    console_hdlr = logging.StreamHandler()
    console_hdlr.setLevel(config.level)
    formatter = logging.Formatter(
        config.fmt.fmt,
        config.fmt.datefmt,
        config.fmt.style,
        config.fmt.validate,
    )
    console_hdlr.setFormatter(formatter)
    logger.addHandler(console_hdlr)
    return logger


def main():
    logger = get_logger()
    logger.info("Hello, World!")
    # log the configuration
    logger.info(f"Logger configuration: {logger.config.to_dict()}")
    # recreate congig
    config = logger.config.to_dict()
    config = LoggingConfig.from_dict(config)
    logger.info(f"Logger configuration: {config.to_dict()}")


if __name__ == "__main__":
    main()
