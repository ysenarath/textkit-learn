import ssl
import warnings

from tklearn.config import config

__version__ = "0.4.0"

__all__ = [
    "__version__",
    "config",
]

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

warnings.filterwarnings(
    "ignore", category=DeprecationWarning, module="datasets.utils._dill"
)
