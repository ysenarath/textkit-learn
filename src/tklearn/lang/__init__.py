from langdetect import detect as detect_lang
from langdetect import detect_langs
from langdetect.lang_detect_exception import LangDetectException

from tklearn.lang import en

__all__ = [
    "en",
    "detect",
    "detect_langs",
    "LangDetectException",
]


def detect(text: str, default: str = "en") -> str:
    try:
        return detect_lang(text)
    except LangDetectException:
        return default
