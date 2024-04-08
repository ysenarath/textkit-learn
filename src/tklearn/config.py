import os
from dataclasses import dataclass, field
from pathlib import Path

from omegaconf import OmegaConf

DEFAULT_CACHE_PATH: str = os.path.abspath(
    os.path.join(os.path.expanduser("~"), ".cache", "tklearn")
)

DEFAULT_CONCEPTNET_DOWNLOAD_URL: str = "https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz"
DEFAULT_NRC_EMONET_DOWNLOAD_URL: str = (
    "https://saifmohammad.com/WebDocs/Lexicons/NRC-Emotion-Lexicon.zip"
)


@dataclass
class ConceptNetConfig:
    download_url: str = f"${{oc.env:TEXTKIT_LEARN_CONCEPTNET_DOWNLOAD_URL,{DEFAULT_CONCEPTNET_DOWNLOAD_URL}}}"


@dataclass
class NRCConfig:
    # National Research Council Canada (NRC)
    emonet_download_url: str = f"${{oc.env:TEXTKIT_LEARN_NRC_EMONET_DOWNLOAD_URL,{DEFAULT_NRC_EMONET_DOWNLOAD_URL}}}"


@dataclass
class ExternalConfig:
    conceptnet: ConceptNetConfig = field(default_factory=ConceptNetConfig)
    nrc: NRCConfig = field(default_factory=NRCConfig)


@dataclass
class Config:
    cache_dir: Path = f"${{oc.env:TEXTKIT_LEARN_CACHE,{DEFAULT_CACHE_PATH}}}"
    external: ExternalConfig = field(default_factory=ExternalConfig)


config: Config = OmegaConf.structured(Config)
