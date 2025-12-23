from __future__ import annotations

from collections.abc import MutableSequence

__all__ = ["FrozenList"]


class FrozenList(MutableSequence):
    def __init__(self, items=None):
        self._items = list(items) if items is not None else []
        self._frozen = False

    @property
    def frozen(self) -> bool:
        return self._frozen

    def freeze(self):
        """Freezes the list, preventing future modifications."""
        self._frozen = True

    def _check_frozen(self):
        """Helper to raise error if list is frozen."""
        if self.frozen:
            raise RuntimeError(
                "Cannot modify a FrozenList after it has been frozen."
            )

    # --- Abstract Methods from MutableSequence ---

    def __getitem__(self, index):
        return self._items[index]

    def __setitem__(self, index, value):
        self._check_frozen()
        self._items[index] = value

    def __delitem__(self, index):
        self._check_frozen()
        del self._items[index]

    def __len__(self):
        return len(self._items)

    def insert(self, index, value):
        self._check_frozen()
        self._items.insert(index, value)

    # --- Hashing and Equality ---

    def __hash__(self):
        if not self._frozen:
            raise TypeError(
                "Cannot hash a FrozenList while it is mutable (unfrozen)."
            )
        # Tuple hashing is efficient and standard for immutable sequences
        return hash(tuple(self._items))

    def __eq__(self, other):
        if isinstance(other, FrozenList):
            return self._items == other._items
        return self._items == other

    def __repr__(self):
        status = "frozen" if self._frozen else "unfrozen"
        return f"<FrozenList(status={status}, items={self._items})>"


def main():
    # --- Example Usage ---
    # Initialize
    fl = FrozenList(["apple", "banana"])
    # Modify
    fl.append("cherry")
    fl[0] = "apricot"
    print(fl)
    # Output: <FrozenList(status=unfrozen, items=['apricot', 'banana', 'cherry'])>
    # Freeze
    fl.freeze()
    # Verify Hashing
    print("Dictionary Key Check: {fl: 'value'}")
    # Works because fl is now hashable
    # Verify Immutability
    try:
        fl.pop()
    except RuntimeError as e:
        print(f"Blocked: {e}")


if __name__ == "__main__":
    main()
