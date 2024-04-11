"""NRC Word-Emotion Association Lexicon.

The NRC Word-Emotion Association Lexicon is a list of words and their 
associations with eight basic emotions (anger, anticipation, disgust, 
fear, joy, sadness, surprise, trust) and two sentiments (negative and 
positive). The lexicon has been used in a variety of research applications, 
including sentiment analysis, emotion recognition, and opinion mining.

Notes
-----
The lexicon is available in two formats: (1) a list of words and their

References
--------
.. [1] “NRC Emotion Lexicon.” Available: 
   https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm. 
   [Accessed: Apr. 11, 2024]

Examples
--------
>>> from tklearn.kb.emolex.io import EmoLexIO
>>> emolex = EmoLexIO()
>>> emolex.download()
>>> for annotation in emolex.load():
...     print(annotation)
"""

from typing import Generator, Optional

import pandas as pd
from tqdm import auto as tqdm

from tklearn import config
from tklearn.base.resource import ResourceIO
from tklearn.utils import download

__all__ = [
    "EmoLexIO",
]


class EmoLexIO(ResourceIO):
    def __init__(self, path: Optional[str] = None):
        if path is None:
            path = config.cache_dir / "resources" / "nrc" / "NRC-Emotion-Lexicon.zip"
        self.path = path
        self.data_path = (
            config.cache_dir
            / "resources"
            / "nrc"
            / "NRC-Emotion-Lexicon"
            / "NRC-Emotion-Lexicon"
            / "NRC-Emotion-Lexicon-ForVariousLanguages.txt"
        )

    def download(
        self,
        url: str = None,
        force: bool = False,
        exist_ok: bool = False,
        unzip: bool = True,
    ):
        """Download NRC Emotion Lexicon from URL"""
        if url is None:
            url = config.external.nrc.emonet_download_url
        try:
            download(
                url,
                self.path,
                force=force,
                exist_ok=exist_ok,
                unzip=unzip,
                mode="wget",
            )
        except ImportError as e:
            raise e
        except Exception:
            try_this_command = f"wget --no-check-certificate -O {self.path} {url}"
            raise ValueError(
                f"failed to get the NRC Emotion Lexicon, try: \n\t{try_this_command}"
            )

    def load(self, verbose: bool = False) -> Generator[dict, None, None]:
        """Load NRC Emotion Lexicon from file"""
        df = pd.read_csv(
            self.data_path,
            delimiter="\t",
        ).rename({"English Word": "English"}, axis=1)
        columns = []
        for col_idx, col_name in enumerate(df.columns):
            if col_idx < 11 and col_idx > 0:
                if col_name in {"negative", "positive"}:
                    col_type = "sentiment"
                else:
                    col_type = "emotion"
            else:
                col_type = "language"
            col = (col_type, col_name)
            columns.append(col)
        df.columns = pd.MultiIndex.from_tuples(columns, names=["type", "name"])
        progess_bar = None
        if verbose:
            progess_bar = tqdm.tqdm(total=len(df), desc="Loading EmoLex")
        for _, row in df.iterrows():
            data = {}
            labels = set()
            for (col_type, col_name), value in row.items():
                if col_type == "language":
                    labels.add((col_name, value))
                    continue
                data.setdefault(col_type, {})[col_name] = float(value)
            for lang, label in labels:
                yield {
                    "label": str(label),
                    "language": str(lang),
                    **data,
                }
            if progess_bar:
                progess_bar.update(1)
        if progess_bar:
            progess_bar.close()
