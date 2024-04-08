from pathlib import Path

import oxrdflib
import pyoxigraph
import rdflib
from tklearn.kb.base import KnowledgeBase

config = {
    "path": "/Users/yasas/Documents/Resources/ConceptNet",
    "version": "5.7",
    "assertions": "conceptnet-assertions-5.7.0.csv",
    "oxigraph": "oxigraph",
}

assertions_path = Path(config["path"]) / config["assertions"]
oxigraph_path = Path(config["path"]) / config["oxigraph"]

assertions_path = Path(config["path"]) / config["assertions"]
oxigraph_path = Path(config["path"]) / config["oxigraph"]


store = oxrdflib.OxigraphStore(store=pyoxigraph.Store(oxigraph_path))

graph = rdflib.Graph(store=store)

kb = KnowledgeBase(graph=rdflib.Graph(store="Oxigraph"))


print(graph.all_nodes())
print("Done")
