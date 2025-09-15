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
    return [{"text": prerix + e["text"]} for e in examples]


def test_group_by():
    ds = Dataset.from_list([
        {"label": i % 10, "text": f"sample {i}"} for i in range(10000)
    ])
    grouped_ds = GroupBy(ds, "label", batch_size=1000).agg(join, prerix=">> ")
    assert grouped_ds.num_rows == 10000, grouped_ds.num_rows


if __name__ == "__main__":
    test_group_by()
