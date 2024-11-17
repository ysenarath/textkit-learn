import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from tklearn import logging

try:
    from IPython.display import HTML, Markdown, display
except ImportError:
    pass

if TYPE_CHECKING:
    from IPython import get_ipython

logger = logging.get_logger(__name__)


def _check_jupyter():
    try:
        # Get the name of the current shell
        shell = get_ipython().__class__.__name__
        if shell in ["ZMQInteractiveShell", "Shell"]:
            return True  # Jupyter notebook or qtconsole
        elif "google.colab" in sys.modules:
            return True  # Google Colab
        else:
            return False  # Terminal Python
    except NameError:
        return False  # Regular Python shell


class DisplayHandler:
    def __init__(self):
        self.is_jupyter = _check_jupyter()
        self.configure_context()

    def configure_context(self):
        """Configure context-specific settings"""
        if self.is_jupyter:
            # Jupyter-specific configurations
            try:
                s = "<style>.container { width:100% !important; }</style>"
                display(HTML(s))
                # Enable inline plotting
                get_ipython().run_line_magic("matplotlib", "inline")
                # Set better figure size defaults for notebooks
                plt.rcParams["figure.figsize"] = [12, 6]
            except ImportError:
                print("Warning: Some Jupyter display features unavailable")
        else:
            # Console-specific configurations
            try:
                # Use non-interactive backend for console
                matplotlib.use("Agg")
            except ImportError:
                print("Warning: matplotlib not available")

    def get_context_info(self):
        info = {
            "is_jupyter": self.is_jupyter,
            "python_version": sys.version,
            "platform": sys.platform,
            "context_type": "Jupyter Notebook"
            if self.is_jupyter
            else "Console",
        }
        # Add additional context-specific information
        if self.is_jupyter:
            try:
                import IPython

                info["ipython_version"] = IPython.__version__
            except ImportError:
                info["ipython_version"] = "Not available"
        return info

    def display(
        self,
        content: Any,
        path: str | Path | None = None,
        markdown: bool = False,
        verbose: bool = False,
        **kwargs,
    ) -> None:
        if isinstance(content, str):
            if self.is_jupyter and markdown:
                display(Markdown(content))
            else:
                print(content)
        elif isinstance(content, (pd.DataFrame, pd.Series)):
            if self.is_jupyter:
                display(content)
            elif path is None:
                print(content)
            else:
                if not isinstance(path, Path):
                    path = Path(path)
                path.parent.mkdir(parents=True, exist_ok=True)
                content.to_csv(path, **kwargs)
                if verbose:
                    logger.info(f"Data saved at: {path}.")
        elif isinstance(content, plt.Figure):
            if self.is_jupyter:
                plt.show()
            else:
                if path is None:
                    path = "output.png"
                if not isinstance(path, Path):
                    path = Path(path)
                path.parent.mkdir(parents=True, exist_ok=True)
                content.savefig(path)
                if verbose:
                    logger.info(f"figure saved at: {path}.")
