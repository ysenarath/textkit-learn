from tklearn.lang.en import _contractions as contractions

__all__ = [
    "contractions",
    "uncontract",
]

uncontract = contractions.expand
