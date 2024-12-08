import re
from pathlib import Path

import pandas as pd

_FLAGS = re.VERBOSE | re.I | re.UNICODE

_contractions_df = pd.read_csv(Path(__file__).with_name("contractions.csv"))


def _build_pattern(df: pd.DataFrame):
    """Build a regex pattern for contractions."""
    cre = {}
    for i, v in df.iterrows():
        c = v["contraction"]
        c = re.escape(c)
        if c in cre:
            continue
        if v["type"] == "word":
            cre[c] = rf"(?:\b(?P<C{i}>" + c + r")\b)"
        elif v["type"] == "suffix":
            cre[c] = rf"(?:(?P<C{i}>" + c + r")\b)"
    return re.compile(r"(?:" + "|".join(cre.values()) + r")", _FLAGS)


_compiled_contraction_regex = _build_pattern(_contractions_df)


def uncontract(text: str):
    def replace(match: re.Match):
        info = _contractions_df.loc[int(match.lastgroup[1:])]
        if info["type"] == "word":
            return info["full_forms"]
        elif info["type"] == "suffix":
            return " " + info["full_forms"]
        return match.group(0)

    return _compiled_contraction_regex.sub(replace, text)
