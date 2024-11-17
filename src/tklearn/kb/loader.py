from __future__ import annotations

from typing import Any, ClassVar, Dict, Iterable, Optional

from nightjar import AutoModule, BaseConfig, BaseModule

__all__ = [
    "AutoKnowledgeLoader",
    "KnowledgeLoader",
    "KnowledgeLoaderConfig",
]


class KnowledgeLoaderConfig(BaseConfig, dispatch="identifier"):
    identifier: ClassVar[str]
    verbose: bool = 1
    namespace: Optional[str] = None
    version: Optional[str] = None


class AutoKnowledgeLoader(AutoModule):
    def __new__(cls, config: KnowledgeLoaderConfig) -> KnowledgeLoader:
        return super().__new__(cls, config)


class KnowledgeLoader(BaseModule):
    config: KnowledgeLoaderConfig

    def download(self) -> None:
        """Download resource."""
        raise NotImplementedError

    def iterrows(self) -> Iterable[Dict[str, Any]]:
        """Iterate over edges."""
        raise NotImplementedError
