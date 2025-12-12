import ssl
import warnings
from contextlib import suppress

import pydantic

from tklearn.config import config


def basic_setup():
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    with suppress(AttributeError):
        warnings.filterwarnings(
            "ignore", category=pydantic.warnings.PydanticDeprecatedSince212
        )
    with suppress(AttributeError):
        warnings.filterwarnings(
            "ignore", category=pydantic.warnings.PydanticDeprecatedSince20
        )


basic_setup()

__version__ = "0.4.0"

__all__ = [
    "__version__",
    "config",
]
