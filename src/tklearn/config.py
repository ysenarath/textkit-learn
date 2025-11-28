import os
from dataclasses import dataclass, field
from pathlib import Path

from omegaconf import OmegaConf

DEFAULT_CACHE_PATH: str = str((Path.home() / ".cache" / "tklearn").absolute())
DEFAULT_DATASET_BATCH_SIZE: int = 1000


def is_debug_enabled() -> bool:
    value = str(os.getenv("TKLEARN_DEBUG", "0")).lower()
    return value in ("1", "true", "yes", "on")


@dataclass
class Config:
    base_dir: Path = f"${{oc.env:TKLEARN_CACHE,{DEFAULT_CACHE_PATH}}}"
    # for cache files only (see utils.cache)
    cache_dir: Path = "${base_dir}/cache"
    # for temp files only (e.g. use with tempfile)
    temp_dir: Path = "${base_dir}/temp"
    # for resources files only (e.g. use with open)
    assets_dir: Path = "${base_dir}/assets"
    # for dataset files only (e.g. use with datasets)
    dataset_batch_size: int = DEFAULT_DATASET_BATCH_SIZE
    # whether to enable debug mode
    debug: bool = field(default_factory=is_debug_enabled)


config: Config = OmegaConf.structured(Config)
