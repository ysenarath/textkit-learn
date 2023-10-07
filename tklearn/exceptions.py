__all__ = [
    "ValidationError",
]


class ValidationError(ValueError):
    ...


class EarlyStoppingException(Exception):
    ...
