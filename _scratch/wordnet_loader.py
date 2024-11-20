from typing import Any, Dict, Iterator, cast
from uuid import uuid4

import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import Lemma, Synset
from tqdm import auto as tqdm


def convert_wordnet_relationships() -> Iterator[Dict[str, Any]]:
    # Ensure WordNet is downloaded
    nltk.download("wordnet", quiet=True)
    # Define common WordNet relations
    relation_mappings = {
        "hypernyms": {"id": "is_a", "label": "Is A", "symmetric": False},
        "hyponyms": {
            "id": "has_instance",
            "label": "Has Instance",
            "symmetric": False,
        },
        "member_meronyms": {
            "id": "has_member",
            "label": "Has Member",
            "symmetric": False,
        },
        "part_meronyms": {
            "id": "has_part",
            "label": "Has Part",
            "symmetric": False,
        },
        "substance_meronyms": {
            "id": "has_substance",
            "label": "Has Substance",
            "symmetric": False,
        },
        "antonyms": {"id": "antonym", "label": "Antonym", "symmetric": True},
    }
    total = 0
    for _ in wn.all_synsets():
        total += 1
    # Process all synsets
    for synset in tqdm.tqdm(wn.all_synsets(), total=total):
        synset = cast(Synset, synset)
        # Create node for the synset
        synset_node = {
            "@id": f"/synset/{synset.name()}",
            "label": synset.name(),
            "language": "en",
            "sense_label": synset.definition(),
            "site": "wordnet",
            "path": f"/synset/{synset.name()}",
            "site_available": True,
        }

        lemma_relation = {
            "@id": "/relation/lemma",
            "label": "Lemma",
            "symmetric": False,
        }

        # Process each lemma in the synset
        for lemma in synset.lemmas():
            # Create node for the lemma
            lemma = cast(Lemma, lemma)
            lemma_node = {
                "@id": f"/lemma/{lemma.name()}",
                "label": lemma.name(),
                "language": "en",
                "term_id": f"/synset/{synset.name()}",
                "site": "wordnet",
                "path": f"/lemma/{lemma.name()}",
                "site_available": True,
            }
            # add lemma node
            edge = {
                "@id": f"/edge/{uuid4()}",
                "rel": lemma_relation,
                "start": synset_node,
                "end": lemma_node,
                "license": "WordNet 3.0",
                "weight": 1.0,
                "dataset": "/dataset",
                "surface_text": None,
                "sources": [
                    {
                        "@id": f"/source/{uuid4()}",
                        "contributor": "WordNet",
                        "process": "wordnet_conversion",
                        "activity": "automated_extraction",
                    }
                ],
            }

        # Process relationships
        for rel_type, rel_info in relation_mappings.items():
            rel_method = getattr(synset, rel_type, None)
            if rel_method is None:
                continue
            related_items = rel_method()
            if not related_items:
                continue
            # Create relation
            relation = {
                "@id": f"/relation/{rel_info['id']}",
                "label": rel_info["label"],
                "symmetric": rel_info["symmetric"],
            }
            # Create edges for each related item
            for related in related_items:
                related = cast(Synset, related)
                edge_id = f"/edge/{uuid4()}"
                edge = {
                    "@id": edge_id,
                    "rel": relation,
                    "start": synset_node,
                    "end": {
                        "@id": f"/synset/{related.name()}",
                        "label": related.name(),
                        "language": "en",
                        "sense_label": related.definition(),
                        "site": "wordnet",
                        "path": f"/synset/{related.name()}",
                        "site_available": True,
                    },
                    "license": "WordNet 3.0",
                    "weight": 1.0,
                    "dataset": "/dataset",
                    "surface_text": f"{synset.name()} {rel_info['label']} {related.name()}",
                    "sources": [
                        {
                            "@id": f"/source/{uuid4()}",
                            "contributor": "WordNet",
                            "process": "wordnet_conversion",
                            "activity": "automated_extraction",
                        }
                    ],
                }
                yield edge


if __name__ == "__main__":
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session
    from tklearn.kb.models import Base, Edge

    engine = create_engine("sqlite:///:memory:")
    session = Session(engine)

    Base.metadata.create_all(engine)

    namespace = "http://wordnetweb.princeton.edu/"

    for edge in convert_wordnet_relationships():
        edge = Edge.from_dict(
            edge, session=session, commit=True, namespace=namespace
        )
        pass
