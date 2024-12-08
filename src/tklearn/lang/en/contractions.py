import re
from pathlib import Path

import pandas as pd

_FLAGS = re.VERBOSE | re.I | re.UNICODE

_contractions_df = pd.read_csv(Path(__file__).with_name("contractions.csv"))


def _build_pattern(df: pd.DataFrame):
    """Build a regex pattern for contractions."""
    cre = []
    for i, v in df.iterrows():
        c = v["contraction"]
        if v["type"] == "word":
            cre.append(rf"(?:\b(?P<C{i}>" + re.escape(c) + r")\b)")
        elif v["type"] == "suffix":
            cre.append(rf"(?:(?P<C{i}>" + re.escape(c) + r")\b)")

    return re.compile(r"(?:" + "|".join(cre) + r")", _FLAGS)


_compiled_contraction_regex = _build_pattern(_contractions_df)


def uncontract(text: str):
    def replace(match: re.Match):
        info = _contractions_df[int(match.lastgroup[1:])]
        if info["type"] == "word":
            return info["full_forms"][0]
        elif info["type"] == "suffix":
            return " " + info["full_forms"][0]
        return match.group(0)

    return _compiled_contraction_regex.sub(replace, text)
