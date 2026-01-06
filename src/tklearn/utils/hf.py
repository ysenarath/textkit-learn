from __future__ import annotations

import contextlib

from huggingface_hub import utils


@contextlib.contextmanager
def suppress_hf_output():
    # Store the original logging verbosity
    prev_verbosity = utils.logging.get_verbosity()
    # Disable progress bars and set logging to ERROR only
    utils.disable_progress_bars()
    utils.logging.set_verbosity_error()
    try:
        yield
    finally:
        # Restore original settings
        utils.enable_progress_bars()
        utils.logging.set_verbosity(prev_verbosity)
