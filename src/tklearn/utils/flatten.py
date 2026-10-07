from __future__ import annotations

import math
import os
from collections.abc import Mapping
from typing import Any

__all__ = [
    "flatten",
    "freeze",
    "unflatten",
]


# (key parts, value) pairs, e.g. (["optim", "lr"], 0.1)
_Entries = list[tuple[list[str], Any]]


class _EmptyMapping(dict):
    # the read-only, hashable {} of `freeze`; a dict, so that json writes
    # it as {} and it equals {}

    __slots__ = ()

    def __hash__(self) -> int:
        return hash(_EmptyMapping)

    def __repr__(self) -> str:
        return "{}"

    def __reduce__(self) -> str:
        return "_EMPTY"  # pickles as the module's one instance

    def __copy__(self) -> _EmptyMapping:
        return self

    def __deepcopy__(self, memo: Any) -> _EmptyMapping:
        return self

    def _read_only(self, *args: Any, **kwargs: Any) -> Any:
        msg = "an empty mapping from flatten is read-only"
        raise TypeError(msg)

    __setitem__ = __delitem__ = __ior__ = _read_only
    clear = pop = popitem = setdefault = update = _read_only


_EMPTY = _EmptyMapping()

# a key that an item of a list lacked, left out of the item
_MISSING = object()


def flatten(values: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten nested values into one level of keys and JSON values.

    Each key of the result can be a column of a table, e.g. to compare
    the settings and results of several runs:

    - nested mappings and named tuples are flattened into dotted keys,
      e.g. ``{"optim": {"lr": 0.1}}`` into ``"optim.lr"``, and a
      `ROCPoints` under ``"roc"`` into ``"roc.fpr"``, ``"roc.tpr"`` and
      ``"roc.thresholds"``; an empty mapping, which has no keys to
      flatten, is kept as a value, ``{}``;
    - lists, tuples and arrays of values are kept whole, as tuples (nested
      for arrays of several dimensions), e.g. per-class scores or a
      confusion matrix;
    - lists holding mappings or named tuples are flattened per key, with
      a tuple over the items, e.g. ``[{"size": 4}, {"size": 8}]`` under
      ``"layers"`` into ``"layers.*.size": (4, 8)``, with None where an
      item lacks the key;
    - the values are made immutable by `freeze`.

    Parameters
    ----------
    values : Mapping
        Values to flatten; keys that are not strings are converted.

    Returns
    -------
    dict
        Keys mapped to frozen values, so a shallow copy is independent of
        it.

    `unflatten` rebuilds the nesting from the result, which is checked:
    ``flatten(unflatten(flat)) == flat``.

    Raises
    ------
    ValueError
        If more than one value flattens to the same key, e.g. with keys
        ``"optim.lr"`` and ``"optim"`` holding ``{"lr": ...}``; or if
        `unflatten` could not rebuild the nesting. That happens when keys
        hold dots that clash with the nesting, e.g. ``"optim"`` and
        ``"optim.lr"`` both holding values, or when the items of a list
        hold a key in different forms that only None tells apart, e.g.
        ``[{"s": None}, {"s": {"a": None}}]`` under ``"x"``, whose keys
        ``"x.*.s"`` and ``"x.*.s.a"`` hold only None.

    Examples
    --------
    >>> flatten({"optim": {"lr": 0.1, "betas": [0.9, 0.99]}})
    {'optim.lr': 0.1, 'optim.betas': (0.9, 0.99)}
    >>> flatten({"layers": [{"size": 4}, {"size": 8, "act": "relu"}]})
    {'layers.*.size': (4, 8), 'layers.*.act': (None, 'relu')}
    """
    flat = _flatten(values)
    _check_reversible(flat)
    return flat


def unflatten(flat: Mapping[str, Any]) -> dict[str, Any]:
    """Rebuild the nested values of `flatten`.

    Dotted keys become nested dicts, and the tuples under ``"*"`` keys a
    list of items, e.g. ``"layers.*.size": (4, 8)`` the list
    ``[{"size": 4}, {"size": 8}]`` under ``"layers"``. The result
    flattens back to `flat` when `flat` comes from `flatten`, which
    checks that it does.

    This loses what `flatten` does not record: named tuples become dicts,
    lists of values stay tuples, and a key that an item of a list lacked
    becomes None in that item.

    Parameters
    ----------
    flat : Mapping
        Flat keys and values, e.g. from `flatten`, or read back from JSON
        that it was written to.

    Returns
    -------
    dict
        The nested values.

    Raises
    ------
    ValueError
        If the keys do not nest, e.g. ``"optim"`` and ``"optim.lr"``, or
        the tuples under a ``"*"`` key differ in length.

    Examples
    --------
    >>> unflatten({"optim.lr": 0.1, "optim.betas": (0.9, 0.99)})
    {'optim': {'lr': 0.1, 'betas': (0.9, 0.99)}}
    >>> unflatten({"layers.*.size": (4, 8), "layers.*.act": (None, "relu")})
    {'layers': [{'size': 4, 'act': None}, {'size': 8, 'act': 'relu'}]}
    """
    return _build([(key.split("."), value) for key, value in flat.items()])


def freeze(value: Any) -> Any:
    """The immutable JSON value that `flatten` stores for a value.

    Lists and tuples become tuples, and an empty mapping a read-only,
    hashable ``{}``. Numpy and torch values become Python values, path
    objects absolute paths (without resolving symlinks, which keeps a
    shared mount as it is; strings are kept as they are), and other
    objects their ``str``.

    Parameters
    ----------
    value : Any
        A value that `flatten` does not flatten further, or a value read
        back from JSON.

    Returns
    -------
    Any
        None, a boolean, number, string or ``{}``, or a tuple of them.
    """
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return tuple(freeze(item) for item in value)
    if isinstance(value, Mapping) and not value:
        return _EMPTY
    if isinstance(value, os.PathLike):
        return os.fsdecode(os.path.abspath(value))
    return str(value)


def _flatten(values: Mapping[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in values.items():
        _flatten_into(flat, str(key), value)
    return flat


def _check_reversible(flat: dict[str, Any]) -> None:
    """Raise a ValueError if `unflatten` would not rebuild `flat`."""
    try:
        nested = unflatten(flat)
    except ValueError as e:
        msg = f"cannot unflatten the flattened values: {e}"
        raise ValueError(msg) from e
    rebuilt = _flatten(nested)
    differ = sorted(
        key
        for key in flat.keys() | rebuilt.keys()
        if key not in flat
        or key not in rebuilt
        or not _same(flat[key], rebuilt[key])
    )
    if differ:
        keys = ", ".join(map(repr, differ))
        msg = (
            f"cannot unflatten the flattened values of {keys}: they would "
            "unflatten to other values, e.g. when the items of a list hold "
            "a key in different forms that only None tells apart"
        )
        raise ValueError(msg)


def _flatten_into(flat: dict[str, Any], key: str, value: Any) -> None:
    if hasattr(value, "tolist"):  # numpy and torch values
        value = value.tolist()
    if (isinstance(value, Mapping) or _is_namedtuple(value)) and value:
        if isinstance(value, Mapping):
            children = value.items()
        else:
            children = zip(value._fields, value)
        for name, child in children:
            _flatten_into(flat, f"{key}.{name}", child)
    elif isinstance(value, (list, tuple)) and any(map(_is_nested, value)):
        _flatten_list(flat, key, value)
    elif _is_namedtuple(value):  # without fields
        _add(flat, key, _EMPTY)
    else:
        _add(flat, key, freeze(value))


def _flatten_list(flat: dict[str, Any], key: str, items: Any) -> None:
    # flatten each item under "key.*", then gather each of their keys into
    # a tuple over the items, with None where an item lacks the key
    flats = []
    for item in items:
        item_flat: dict[str, Any] = {}
        _flatten_into(item_flat, f"{key}.*", item)
        flats.append(item_flat)
    for name in dict.fromkeys(k for f in flats for k in f):
        _add(flat, name, tuple(f.get(name) for f in flats))


def _add(flat: dict[str, Any], key: str, value: Any) -> None:
    if key in flat:
        msg = f"more than one value flattens to the key {key!r}"
        raise ValueError(msg)
    flat[key] = value


def _is_namedtuple(value: Any) -> bool:
    return isinstance(value, tuple) and hasattr(value, "_fields")


def _is_nested(value: Any) -> bool:
    """Whether a value holds a mapping or named tuple, which `freeze`
    cannot keep."""
    if isinstance(value, Mapping) or _is_namedtuple(value):
        return True
    return isinstance(value, (list, tuple)) and any(map(_is_nested, value))


def _any_set(value: Any, depth: int) -> bool:
    # whether a value is not None, inside `depth` levels of tuples over
    # the items of lists
    if value is None:
        return False
    if depth == 0 or not isinstance(value, (list, tuple)):
        return True
    return any(_any_set(item, depth - 1) for item in value)


def _join(*parts: str) -> str:
    return ".".join(part for part in parts if part)


def _same(a: Any, b: Any) -> bool:
    # equality of frozen values, with NaN equal to NaN
    if isinstance(a, tuple) and isinstance(b, tuple):
        return len(a) == len(b) and all(map(_same, a, b))
    if isinstance(a, float) and isinstance(b, float):
        return a == b or (math.isnan(a) and math.isnan(b))
    return a == b


def _build(
    entries: _Entries,
    prefix: str = "",
    unset: frozenset[str] | None = None,
    rel: str = "",
) -> dict[str, Any]:
    # `unset` and `rel` are given inside an item of a list: the keys that
    # no item of the list sets, and the key of `entries`, relative to an
    # item
    groups: dict[str, _Entries] = {}
    for parts, value in entries:
        groups.setdefault(parts[0], []).append((parts[1:], value))
    result: dict[str, Any] = {}
    for name, group in groups.items():
        key = f"{prefix}{name}"
        value = _build_value(key, group, unset, _join(rel, name))
        if value is not _MISSING:
            result[name] = value
    return result


def _build_value(
    key: str, entries: _Entries, unset: frozenset[str] | None, rel: str
) -> Any:
    # the forms `key` can take: its own value (no parts left), list items
    # ("*") or nested keys
    own = [(rest, v) for rest, v in entries if not rest]
    items = [(rest, v) for rest, v in entries if rest and rest[0] == "*"]
    nested = [(rest, v) for rest, v in entries if rest and rest[0] != "*"]
    forms = [form for form in (own, items, nested) if form]
    if len(forms) > 1 and unset is not None:
        # items of a list can differ, e.g. [{"a": 1}, {"a": {"b": 2}}];
        # there, None stands for what an item lacked, so the item has the
        # form that is set
        set_forms = [f for f in forms if any(v is not None for _, v in f)]
        if not set_forms:
            # nothing is set, so this item may hold a key that no item of
            # the list sets: take the form that has one
            own_unset = bool(own) and rel in unset
            if not own_unset and any(
                _join(rel, *rest) in unset for rest, _ in nested
            ):
                return _build_nested(nested, key, unset, rel)
            return None if own else _MISSING
        forms = set_forms
    if len(forms) > 1:
        msg = (
            f"{key!r} holds more than one of a value, list items and nested "
            "keys"
        )
        raise ValueError(msg)
    (form,) = forms
    if form is own:
        value = own[0][1]
        # a new dict, like the other mappings of the result
        return {} if isinstance(value, Mapping) and not value else value
    if form is items:
        return _build_list([(rest[1:], v) for rest, v in items], key)
    return _build_nested(nested, key, unset, rel)


def _build_nested(
    entries: _Entries, key: str, unset: frozenset[str] | None, rel: str
) -> Any:
    nested = _build(entries, f"{key}.", unset, rel)
    # in an item, a mapping without the keys that the item lacked is
    # lacked too; an empty mapping is a value, {}
    return _MISSING if not nested and unset is not None else nested


def _build_list(entries: _Entries, key: str) -> Any:
    # each entry holds a tuple with a value per item, or None when an item
    # of an enclosing list lacked this list
    if all(value is None for _, value in entries):
        return _MISSING
    lengths = {
        len(value) if isinstance(value, (list, tuple)) else -1
        for _, value in entries
        if value is not None
    }
    if len(lengths) != 1 or -1 in lengths:
        msg = f"the values under '{key}.*' must be tuples of one length"
        raise ValueError(msg)
    (length,) = lengths
    # the keys, relative to an item, that no item of this list sets
    unset = frozenset(
        ".".join(rest)
        for rest, value in entries
        if not _any_set(value, 1 + rest.count("*"))
    )
    items = []
    for i in range(length):
        # an entry of None: the enclosing item lacked these keys entirely
        item = [(rest, v[i]) for rest, v in entries if v is not None]
        value = _build_value(f"{key}.*", item, unset, rel="")
        # nothing set in a mapping item: one without keys
        items.append({} if value is _MISSING else value)
    return items
