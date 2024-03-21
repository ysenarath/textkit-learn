from collections.abc import Mapping, Sequence
from typing import Any, Dict, Generator, List

__all__ = [
    "flatten",
    "list_of_dicts_to_dict_of_lists",
    "rebuilt",
]

NA = object()


def dict_key_escape(s: str) -> str:
    """
    Escape dots in a string.

    Parameters
    ----------
    s : str
        The string to escape.

    Returns
    -------
    str
        The escaped string.
    """
    return s.replace("\\", "\\\\").replace(".", "\\.")


def dict_key_parse(s: str) -> Generator[str, None, None]:
    """
    Parse a string into a list of keys by splitting on dots, but not on escaped dots.

    Parameters
    ----------
    s : str
        The string to parse.

    Yields
    ------
    str
        The keys.
    """
    skip_next = False
    current = None
    for c in s:
        if current is None:
            current = ""
        current += c
        if skip_next:
            skip_next = False
        elif c == "\\":
            current = current[:-1]
            skip_next = True
        elif c == ".":
            yield current[:-1]
            current = None
    if current is not None:
        yield current


def flatten(
    d: Mapping,
) -> Mapping:
    """
    Normalize a nested dictionary by flattening it.

    Parameters
    ----------
    d : dict
        The dictionary to normalize.
    order : ReturnOrder, optional
        The order of the returned dictionary, by default ReturnOrder.original

    Returns
    -------
    dict
        The normalized dictionary.
    """
    out = {}
    for key, value in d.items():
        key = dict_key_escape(key)
        value = flatten(value)
        if isinstance(value, Mapping):
            for k, v in value.items():
                out[f"{key}.{k}"] = v
        else:
            out[key] = value
    return out


def rebuilt(d: Mapping) -> Mapping:
    out = {}
    for key, value in d.items():
        key = tuple(dict_key_parse(key))
        current = out
        for k in key[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        current[key[-1]] = value
    return out


def list_of_dicts_to_dict_of_lists(d: Sequence[Mapping]) -> Mapping:
    out: Dict[Any, List] = {}
    if isinstance(d, str) or not isinstance(d, Sequence):
        msg = "expected a list"
        raise ValueError(msg)
    for i in d:
        for k, _ in i.items():
            if k not in out:
                out[k] = []
    for i in d:
        if not isinstance(i, Mapping):
            msg = "expected a list of dicts"
            raise ValueError(msg)
        for k in out:
            v = i.get(k, NA)
            out[k].append(v)
    return out
