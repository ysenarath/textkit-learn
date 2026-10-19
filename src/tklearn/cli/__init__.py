"""The ``tklearn`` command.

Each group of commands is a module of this package, added to `main`
here: ``tklearn runs`` (`tklearn.cli.runs`) follows the runs recorded by
`tklearn.tracing.FileTracerProvider`, and ``tklearn kb``
(`tklearn.cli.kb`) queries a knowledge base. Group modules import only
click and rich at the top, and the rest inside their commands, so that
``tklearn --help`` stays fast.
"""

from __future__ import annotations

import click

import tklearn
from tklearn.cli.kb import kb
from tklearn.cli.runs import runs

__all__ = [
    "main",
]


@click.group()
@click.version_option(tklearn.__version__, prog_name="tklearn")
def main() -> None:
    """Tools of textkit-learn."""


main.add_command(kb)
main.add_command(runs)
