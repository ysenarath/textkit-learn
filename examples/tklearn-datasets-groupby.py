from collections.abc import Hashable

from datasets import Dataset

from tklearn.utils.datasets import GroupBy


def create_groups_indices(
    keys: list[Hashable], idxs: list[int], groups: dict[Hashable, list[int]]
):
    for key, i in zip(keys, idxs):
        groups[key].append(i)


def join(examples, prerix=None):
    if prerix is None:
        prerix = ""
    return {"text": [prerix + " ".join(examples["text"])]}


def test_group_by():
    ds = Dataset.from_dict({
        "id": [0, 1, 2, 3, 4, 5],
        "text": [
            "Hello world",
            "How are you?",
            "Fine, thanks.",
            "Goodbye!",
            "See you later.",
            "Take care.",
        ],
        "label": [0, 0, 0, 1, 1, 1],
    })
    grouped_ds = GroupBy(ds, "label").agg(join, prerix=">> ")
    print(grouped_ds[:])
    assert grouped_ds.num_rows == 2


if __name__ == "__main__":
    test_group_by()
