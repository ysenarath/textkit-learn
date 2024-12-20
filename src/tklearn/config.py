from dataclasses import dataclass
from pathlib import Path

from omegaconf import OmegaConf

DEFAULT_CACHE_PATH: str = str((Path.home() / ".cache" / "tklearn").absolute())


@dataclass
class Config:
    cache_dir: Path = f"${{oc.env:TKLEARN_CACHE,{DEFAULT_CACHE_PATH}}}"


config: Config = OmegaConf.structured(Config)
