from dataclasses import dataclass, field
from pathlib import Path

from omegaconf import OmegaConf

DEFAULT_CACHE_PATH: str = str((Path.home() / ".cache" / "tklearn").absolute())


@dataclass
class ConceptNetConfig:
    download_url: str = "${{oc.env:{env_var},{url}}}".format(
        env_var="TKLEARN_CONCEPTNET_DOWNLOAD_URL",
        url="https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz",
    )


@dataclass
class NRCConfig:
    # National Research Council Canada (NRC)
    emonet_download_url: str = "${{oc.env:{env_var},{url}}}".format(
        env_var="TKLEARN_NRC_EMONET_DOWNLOAD_URL",
        url="https://saifmohammad.com/WebDocs/Lexicons/NRC-Emotion-Lexicon.zip",
    )


@dataclass
class ExternalConfig:
    conceptnet: ConceptNetConfig = field(default_factory=ConceptNetConfig)
    nrc: NRCConfig = field(default_factory=NRCConfig)


@dataclass
class Config:
    cache_dir: Path = f"${{oc.env:TKLEARN_CACHE,{DEFAULT_CACHE_PATH}}}"
    external: ExternalConfig = field(default_factory=ExternalConfig)


config: Config = OmegaConf.structured(Config)
