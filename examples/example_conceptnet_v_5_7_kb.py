# import rdflib
# from tklearn.config import config

# url = config.external.conceptnet.download_url
# output_dir = config.cache_dir / "resources" / "conceptnet"
# filename = url.split("/")[-1]

# jsonld_file = (output_dir / filename).with_suffix("").with_suffix(".jsonld")

# if not jsonld_file.exists():
#     raise FileNotFoundError(f"File not found: {jsonld_file}")

# graph = rdflib.Graph().parse(jsonld_file, format="json-ld")

# print(len(graph))

# from rdflib import Graph

# graph = Graph(store="BerkeleyDB")

# import plyvel

# with plyvel.DB("/tmp/testdb/", create_if_missing=True) as db:
#     db.put(b"key", b"value")
#     print(db.get(b"key"))
#     db.delete(b"key")
