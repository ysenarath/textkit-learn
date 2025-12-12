import ssl
import warnings

from tklearn.config import config


def basic_setup():
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    # Suppress all Pydantic deprecation warnings coming from libraries like MLflow
    warnings.filterwarnings("ignore", module="pydantic")
    warnings.filterwarnings("ignore", module="mlflow")
    # Suppress the specific regex escape warning
    warnings.filterwarnings(
        "ignore",
        category=DeprecationWarning,
        message="invalid escape sequence",
    )


basic_setup()

__version__ = "0.4.0"

__all__ = [
    "__version__",
    "config",
]
