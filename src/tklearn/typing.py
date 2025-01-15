from __future__ import annotations

__all__ = [
    "UNDEFINED",
]


class Undefined:
    # undefined indicates the absence of a value

    instance: Undefined = None

    @classmethod
    def get_instance(cls) -> Undefined:
        if cls.instance is None:
            cls.instance = super().__new__(cls)
        return cls.instance

    def __new__(cls):
        return cls.get_instance()

    def __repr__(self):
        return "NULL"


UNDEFINED = Undefined()
