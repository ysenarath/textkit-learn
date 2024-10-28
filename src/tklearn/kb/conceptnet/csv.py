import orjson

from tklearn.kb.conceptnet.uri import conjunction_uri, to_json_ld

__all__ = [
    "read_line",
]


def read_line(line: str) -> dict:
    parts = line.strip().split("\t")
    edge_id, relation_uri, subject_uri, object_uri, metadata = parts
    metadata = orjson.loads(metadata)
    edge = {
        "@id": edge_id,
        "rel": to_json_ld(relation_uri),
        "start": to_json_ld(subject_uri),
        "end": to_json_ld(object_uri),
    }
    if "surfaceText" in metadata:
        edge["surfaceText"] = metadata["surfaceText"]
    if "sources" in metadata:
        sources = []
        source: dict
        for source in metadata["sources"]:
            source["@id"] = conjunction_uri(*source.values())
            sources.append(source)
        edge["sources"] = sources
    if "license" in metadata:
        edge["license"] = metadata["license"]
    if "weight" in metadata:
        edge["weight"] = metadata["weight"]
    if "dataset" in metadata:
        edge["dataset"] = metadata["dataset"]
    return edge
