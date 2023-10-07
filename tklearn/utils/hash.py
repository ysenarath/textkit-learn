import functools
import inspect

import pickle
import joblib

__all__ = [
    "hash",
]


def create_hashable(obj):
    if isinstance(obj, dict):
        return tuple(sorted((k, create_hashable(v)) for k, v in obj.items()))
    elif isinstance(obj, (list, tuple)):
        return tuple(create_hashable(v) for v in obj)
    elif isinstance(obj, functools.partial):
        func = obj.func
        try:
            s = inspect.signature(func)
            bound = s.bind_partial(*obj.args, **obj.keywords)
            bound.apply_defaults()
            hashable_args = create_hashable(bound.arguments)
        except ValueError:
            # ignores the name of the function (what matters is the content)
            hashable_args = (
                create_hashable(obj.args),
                create_hashable(obj.keywords),
            )
        if isinstance(func, functools.partial):
            hashable_func = create_hashable(func)
        else:
            hashable_func = inspect.getsource(func)
            # hashable_func = pickle.dumps(func)
        return hashable_func, hashable_args
    elif callable(obj):
        return create_hashable(functools.partial(obj))
    return obj


def hash(obj):
    obj = create_hashable(obj)
    return joblib.hash(obj)
