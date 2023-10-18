from pathlib import Path

from tklearn.config import configurable

__all__ = [
    "resources",
]


@configurable(name="resource")
class ResourceManager(object):
    def __init__(self, path):
        self.path = Path(path)


resources = ResourceManager()
