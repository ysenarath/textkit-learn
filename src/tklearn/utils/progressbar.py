from __future__ import annotations

import contextlib
import functools
import io
import json
import sys
import weakref
from typing import Any, Iterable, List, Optional, TypeVar, Union

import pandas as pd
from IPython.display import display as ipy_display
from ipywidgets import widgets
from tabulate import tabulate
from tqdm.auto import tqdm as auto_tqdm
from tqdm.notebook import tqdm as nb_tqdm

__all__ = [
    "ProgressBar",
]

T = TypeVar("T")

table_css = """<style scoped>
    .progressbar-table {
        font-family: Arial, sans-serif;
        border-collapse: collapse;
        width: 100%;
        overflow-x: auto;
    }

    .progressbar-table th, .progressbar-table td {
        padding: 8px;
        text-align: left;
        border-bottom: 1px solid #ddd;
    }

    .progressbar-table tr:nth-child(even) {
        background-color: #f2f2f2;
    }

    .progressbar-table th {
        background-color: #4CAF50;
        color: white;
    }
</style>"""


class ProgressBarTable:
    def __init__(self, pbar: Optional[ProgressBar] = None):
        self.get_pbar = weakref.ref(pbar)
        self.data = {}

    @property
    def container(self) -> str:
        return widgets.HTML(self.to_html())

    def add_row(self, *args, **kwargs) -> None:
        row = dict(*args, **kwargs)
        prow = {}
        for key, value in row.items():
            # make sure the data is json serializable
            with contextlib.suppress(TypeError):
                prow[key] = json.loads(json.dumps(value))
        num_rows = self.num_rows
        keys = set(self.data.keys()).union(row.keys())
        table_data: dict[str, List[Optional[Any]]] = {}
        for key in keys:
            value = row.get(key)
            if key in self.data:
                table_data[key] = self.data[key].copy()
            else:
                table_data[key] = [None] * num_rows
            table_data[key].append(value)
        self.data = table_data
        pbar = self.get_pbar()
        pbar.refresh()

    @property
    def num_rows(self) -> int:
        return len(next(iter(self.data.values()))) if self.data else 0

    def to_string(self) -> str:
        return tabulate(self.data, headers="keys", tablefmt="psql")

    def to_html(self) -> str:
        table = pd.DataFrame(self.data).to_html(
            index=False, classes="progressbar-table", border=1
        )
        return f"<div>{table_css}{table}</div>"


class ProgressBarIO(io.StringIO):
    def __init__(self, pbar: Optional[ProgressBar] = None):
        super().__init__()
        self.get_pbar = weakref.ref(pbar)
        self.current_line = ""

    def write(self, s: str) -> int:
        return super().write(s)

    def flush(self) -> None:
        super().flush()
        self.current_line = self.getvalue().strip()
        self.truncate(0)
        self.seek(0)
        self.get_pbar().refresh()

    def display(self) -> None:
        pbar = self.get_pbar()
        content = self.current_line + "\r"
        if pbar and pbar.table:
            content += "\n\n"
            content += pbar.table.to_string() + "\n"
        sys.stdout.write(content.strip())


class ProgressBar(Iterable[T]):
    def __init__(self, *args, **kwargs):
        self.table = ProgressBarTable(self)
        if nb_tqdm in auto_tqdm.__bases__:
            bound_tqdm = functools.partial(auto_tqdm, *args, **kwargs)
            # display = bound_tqdm.keywords.pop("display", False)
            bound_tqdm.keywords["display"] = False
            self.tqdm = bound_tqdm()
            container = widgets.VBox([])
            ipy_display(container)
            self.container = container
            self.refresh()
        else:
            container = ProgressBarIO(self)
            # display = kwargs.pop("display", False)
            # kwargs.update({"file": container, "ascii": False})
            self.tqdm = auto_tqdm(*args, **kwargs)
            self.container = container

    def __iter__(self) -> Iterable[T]:
        yield from self.tqdm

    def update(self, n: Union[int, float] = 1) -> None:
        self.tqdm.update(n)

    def set_postfix(self, *args, **kwargs) -> None:
        self.tqdm.set_postfix(*args, **kwargs)

    def set_description_str(
        self, desc: Optional[str] = None, refresh: Optional[bool] = True
    ) -> None:
        self.tqdm.set_description_str(desc, refresh=refresh)

    def close(self) -> None:
        self.tqdm.close()

    def refresh(self) -> None:
        if isinstance(self.container, ProgressBarIO):
            # display the progress bar and table (replace or print)
            self.container.display()
        elif self.table.num_rows > 0:
            # update the table
            self.container.children = [
                self.tqdm.container,
                self.table.container,
            ]
        else:
            self.container.children = [
                self.tqdm.container,
            ]
