from typing import Any, Generator

__all__ = [
    "ResourceIO",
]


class ResourceIO:
    def __init__(self, path: str):
        self.path = path

    def download(self, *args: Any, **kwargs: Any) -> None:
        """Download resource."""
        raise NotImplementedError

    def load(self, *args: Any, **kwargs: Any) -> Generator[dict, None, None]:
        """Load resource."""
        raise NotImplementedError
