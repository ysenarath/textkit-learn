import io
import sys
from typing import Iterable, Optional, TypeVar, Union

from tabulate import tabulate

T = TypeVar("T")


class ProgressBar(Iterable[T]):
    def __init__(self, total: int, desc: Optional[str] = None, **kwargs):
        self.total = total
        self.desc = desc
        self.n = 0
        self.postfix = {}
        self.table_data = []
        self.output = io.StringIO()

    def __iter__(self) -> Iterable[T]:
        for i in range(self.total):
            yield i
            self.update(1)

    def update(self, n: Union[int, float] = 1) -> None:
        self.n += n
        self.refresh()

    def set_postfix(self, **kwargs) -> None:
        self.postfix.update(kwargs)
        self.refresh()

    def set_description_str(self, desc: Optional[str] = None) -> None:
        self.desc = desc
        self.refresh()

    def add_row(self, row: dict) -> None:
        self.table_data.append(row)
        self.refresh()

    def refresh(self) -> None:
        self.output.truncate(0)
        self.output.seek(0)

        # Progress bar
        bar_length = 50
        filled_length = int(self.n / self.total * bar_length)
        bar = "█" * filled_length + "-" * (bar_length - filled_length)
        percent = f"{100 * self.n / self.total:.1f}%"

        progress_line = (
            f"\r{self.desc or ''}: |{bar}| {self.n}/{self.total} [{percent}]"
        )
        if self.postfix:
            postfix_str = ", ".join(f"{k}={v}" for k, v in self.postfix.items())
            progress_line += f" {postfix_str}"

        self.output.write(progress_line)

        # Table
        if self.table_data:
            self.output.write("\n\n")
            table = tabulate(self.table_data, headers="keys", tablefmt="grid")
            self.output.write(table)

        # Print to console
        sys.stdout.write(self.output.getvalue())
        sys.stdout.write("\n")  # Add a newline for better readability
        sys.stdout.flush()

    def close(self) -> None:
        self.refresh()
        sys.stdout.write("\n")  # Final newline
        sys.stdout.flush()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# Example usage
if __name__ == "__main__":
    import time

    with ProgressBar(total=100, desc="Processing") as pbar:
        for i in range(100):
            time.sleep(0.1)  # Simulate some work
            pbar.update(1)
            if i % 20 == 0:
                pbar.add_row({"Step": i, "Value": i * 2})
            if i % 25 == 0:
                pbar.set_postfix(current=i)
