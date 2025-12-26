from __future__ import annotations

from collections import defaultdict

import graphviz
import orjson
from graphviz import escape

from tklearn.utils.frozenlist import FrozenList

__all__ = [
    "plot_token_tree",
    "TokenTree",
]


def _generate_node_label(node: dict, index: int = 0) -> str:
    label = escape(node["label"])
    hard_index_key = "index"
    if "hard_index" in node:
        hard_index_key = "hard_index"
    hard_idx = node.get(hard_index_key) or index
    soft_index_key = "depth"
    if "soft_index" in node:
        soft_index_key = "soft_index"
    soft_idx = node.get(soft_index_key) or None
    prefix, suffix = "", ""
    if hard_idx is not None:
        prefix = f'<TR><TD><FONT FACE="Times-Roman" COLOR="black" POINT-SIZE="12"><B>{hard_idx}</B></FONT></TD></TR>'
    if soft_idx is not None:
        suffix = f'<TR><TD><FONT FACE="Times-Roman" COLOR="red" POINT-SIZE="12">{soft_idx}</FONT></TD></TR>'
    return f"""<
    <TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="1">
        {prefix}<TR><TD><FONT FACE="Times-Roman" POINT-SIZE="18"><B>{label}</B></FONT></TD></TR>{suffix}
    </TABLE>
>"""


def plot_token_tree(
    nodes: list[dict], backbone: list[int]
) -> graphviz.Digraph:
    dot = graphviz.Digraph(comment="Sentence Tree")
    dot.attr(rankdir="LR")
    dot.attr("node", shape="none", fontname="Helvetica")

    # Track which nodes have been created inside clusters so we don't duplicate them
    processed_indices = set()

    idx_to_token = {i: token for i, token in enumerate(nodes)}

    groups: dict[str, list[int]] = defaultdict(list)
    for i, token in enumerate(nodes):
        if "group" in token:
            group_name = token["group"]
            groups[group_name].append(i)
    groups = dict(groups)

    # 1. Create Clusters (The Boxes)
    for i, (_, group_indices) in enumerate(groups.items()):
        # Name must start with 'cluster' to be visible as a box
        with dot.subgraph(name=f"cluster_{i}") as c:
            c.attr(style="dashed", color="black", label="")  # Box styling
            for idx in group_indices:
                processed_indices.add(idx)
                # Find the token data for this index
                token = idx_to_token.get(idx, None)
                if token:
                    label_html = _generate_node_label(token, index=idx)
                    # Maintain the spine alignment logic inside the cluster
                    if idx in backbone:
                        c.node(str(idx), label=label_html, group="main")
                    else:
                        c.node(str(idx), label=label_html)

    # 2. Add Remaining Nodes (Those not in any box)
    for idx, token in enumerate(nodes):
        if idx not in processed_indices:
            label_html = _generate_node_label(token, index=idx)
            if idx in backbone:
                dot.node(str(idx), label=label_html, group="main")
            else:
                dot.node(str(idx), label=label_html)

    # 3. Add Edges with 'weight'
    for idx, token in enumerate(nodes):
        if "parent" in token:
            parent_idx = token["parent"]
        elif idx > 0:
            parent_idx = idx - 1
        else:
            parent_idx = None
        if isinstance(parent_idx, str):
            # Find the first index of the group with this name
            group_indices = groups[parent_idx]
            parent_idx = group_indices[-1]
        if parent_idx is None:
            continue
        # Logic for edge thickness/straightness
        if parent_idx in backbone and idx in backbone:
            dot.edge(str(parent_idx), str(idx), weight="10")
        else:
            dot.edge(str(parent_idx), str(idx), weight="1")

    # output_path = dot.render(filename, view=True, format="png")
    return dot


def _search_path(
    *,
    nodes: list[dict],
    start_index: int | None,
    end_index: int | None,
    groups: dict[str, list[int]] = {},
) -> list[int]:
    if start_index is None or end_index is None:
        return []
    path = [end_index]
    current_index = end_index
    while current_index != start_index:
        try:
            current_node = nodes[current_index]
        except TypeError:
            raise TypeError(", ".join(map(str, path)))
        if "parent" in current_node:
            parent = current_node["parent"]
        else:
            parent = current_index - 1
        if not isinstance(parent, str):
            parent_index = parent
        else:
            # find the first occurrence of the group
            parent_indices = groups.get(parent, [])
            parent_index = parent_indices[-1] if parent_indices else None
        # add parent_index to path until we reach start_index
        path.insert(0, parent_index)
        current_index = parent_index
    return path


class TokenTree:
    def __init__(self):
        self.nodes = FrozenList()
        self.id2index = {}
        self.head = None
        self.tail = None
        self.groups: dict[str, list[int]] | None = None

    def dumps(self) -> dict:
        data = {
            "nodes": list(self.nodes),
            "id2index": dict(self.id2index),
            "head": self.head,
            "tail": self.tail,
            "groups": self.groups,
        }
        return orjson.dumps(
            data,
            option=orjson.OPT_SERIALIZE_NUMPY
            | orjson.OPT_SORT_KEYS
            | orjson.OPT_NON_STR_KEYS,
        ).decode()

    @classmethod
    def loads(cls, s: str):
        self = cls.__new__(cls)
        data = orjson.loads(s.encode())
        self.nodes = FrozenList(data["nodes"])
        self.id2index = data["id2index"]
        self.head = data["head"]
        self.tail = data["tail"]
        self.groups = data["groups"]
        return self

    def __getitem__(self, index):
        return self.nodes[index]

    def __iter__(self):
        return iter(self.nodes)

    def __len__(self):
        return len(self.nodes)

    def lookup(self, id: int | str, **kwargs) -> int:
        if isinstance(id, int):
            return id
        if "default" in kwargs:
            default = kwargs["default"]
            return self.id2index.get(str(id), default)
        return self.id2index[str(id)]

    def append(self, label: str, id: str = None, **kwargs) -> int:
        # kwargs: parent, depth, soft_index, hard_index, group
        idx = len(self.nodes)
        self.nodes.append({"label": label, **kwargs})
        if id:
            id = id.format(index=idx, label=label)
            self.id2index[str(id)] = len(self.nodes) - 1
        return idx

    def build(self):
        """Calculate soft indices for words without depth defined."""
        self.groups = {}
        for i, info in enumerate(self.nodes):
            if "group" in info:
                group = info["group"]
                if group not in self.groups:
                    self.groups[group] = []
                self.groups[group].append(i)
        node_to_depth = {}
        current_depth = 0
        for i, info in enumerate(self.nodes):
            if "depth" in info:
                node_to_depth[i] = info["depth"]
            else:
                parent_index_found_from = ""
                if "parent" in info:
                    parent = info["parent"]
                    parent_index_found_from = " explicit"
                else:
                    parent = i - 1
                    parent_index_found_from = " implicit"
                # parent can be an index or a group
                if isinstance(parent, str):
                    # find the first occurrence of the group
                    parent_indices = self.groups.get(parent, [])
                    parent_index = (
                        parent_indices[-1] if parent_indices else None
                    )
                    parent_index_found_from += " group"
                else:
                    parent_index = parent
                    parent_index_found_from += " index"
                if parent_index is None:
                    info["depth"] = current_depth
                    current_depth += 1
                elif parent_index >= 0:
                    info["depth"] = node_to_depth[parent_index] + 1
                else:
                    info["depth"] = current_depth
                    current_depth += 1
                node_to_depth[i] = info["depth"]
        self.nodes.freeze()

    def graphviz(self) -> graphviz.Digraph:
        if not self.nodes.frozen:
            raise ValueError("TokenTree must be built before plotting.")
        start_index = self.lookup(self.head, default=None)
        end_index = self.lookup(self.tail, default=None)
        backbone = _search_path(
            nodes=self.nodes,
            start_index=start_index,
            end_index=end_index,
            groups=self.groups,
        )
        return plot_token_tree(self.nodes, backbone)
