from dataclasses import dataclass
from pathlib import Path

from omegaconf import OmegaConf

DEFAULT_CACHE_PATH: str = str((Path.home() / ".cache" / "tklearn").absolute())
DEFAULT_DATASET_BATCH_SIZE: int = 1000


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


config: Config = OmegaConf.structured(Config)
