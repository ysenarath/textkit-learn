from __future__ import annotations

import gzip
import os
import pickle
import shutil
import subprocess
from pathlib import Path
from typing import ClassVar, Union

import numpy as np
import requests
from huggingface_hub import HfApi
from setfit import SetFitModel
from tqdm import auto as tqdm

from tklearn import config, logging
from tklearn.embeddings.base import AutoEmbedding
from tklearn.kb.base import ArtifactStore, ArtifactStoreConfig
from tklearn.kb.lexicon import Lexicon
from tklearn.kb.triple_store import TripleStore
from tklearn.kb.wiktionary.models import Word, parse_jsonl

logger = logging.get_logger(__name__)


_WIKTIONARY_URL = "https://kaikki.org/dictionary/raw-wiktextract-data.jsonl.gz"


def normalize(a: str, b: str, c: str) -> tuple[str, str, str]:
    # convert to bottom-up relation if needed
    if b == "hyponym" or b == "instance":
        # here the original relation is top-down
        #   i.e., c is hyponym of a / c is instance of a
        return (c, b, a)
    if b == "hypernym":
        # c is hypernym of a (top-down, i.e., c is at the top)
        b = "hyponym"
        # a is hyponym of c (bottom-up)
        return (a, b, c)
    if b == "synonym" or b == "antonym":
        # symmetrical (not DAG)
        # a is synonym of b == b is synonym of a
        return (a, b, c)
    raise ValueError(f"relation type {b!r} is not recognized")


def count_jsonl(path: Path) -> int:
    return int(subprocess.check_output(f"wc -l {path}", shell=True).split()[0])


def add_forms(w: Word, lexicon: Lexicon):
    """Stores form mappings in LevelDB using batch writing."""
    word = (w.word or "").strip()
    if not word:
        return
    base_forms: set = lexicon.get(word, {word})
    for form_of in w.form_of or []:
        form_of_word = (form_of.word or "").strip()
        base_forms.add(form_of_word)
    lexicon[word] = base_forms
    for form in w.forms or []:
        form_form = (form.form or "").strip()
        if not form_form:
            continue
        base_forms: set = lexicon.get(form_form, set())
        base_forms.add(word)
        lexicon[form_form] = base_forms


class WiktionaryArtifactStoreConfig(ArtifactStoreConfig):
    name: ClassVar[str] = "wiktionary"
    repo_id: str = "textkit-learn/wiktionary"
    repo_type: str = "dataset"
    private: bool = True


class WiktionaryArtifactStore(ArtifactStore):
    config: WiktionaryArtifactStoreConfig

    def __post_init__(self):
        # local_dir is the directory where the repo will be downloaded
        self.local_dir = Path(config.assets_dir) / "wiktionary"
        self.local_dir.mkdir(parents=True, exist_ok=True)
        # repo_id is the name of the repo on Hugging Face Hub
        self.repo_id = self.config.repo_id
        self.repo_type = self.config.repo_type
        self.private = self.config.private
        # must be compatible with wiktextract library (https://kaikki.org)
        self.wiktionary_url = _WIKTIONARY_URL
        # Hugging Face Hub API
        self.api = HfApi()
        # config
        self.predicates = [
            "synonym",
            "antonym",
            "hyponym",
            "hypernym",
            "instance",
        ]
        self.lang_code: str = "en"
        # create the repo if it does not exist
        self.api.create_repo(
            repo_id=self.repo_id,
            private=self.private,
            repo_type=self.repo_type,
            exist_ok=True,
        )
        # download the repo to the local directory
        self.api.snapshot_download(
            repo_id=self.repo_id,
            repo_type=self.repo_type,
            local_dir=self.local_dir,
        )
        filename = "wiktionary.jsonl"
        lang = self.lang_code
        self.download_path = self.local_dir / f"{filename}.gz"
        self.extracted_path = self.local_dir / filename
        self.gloss2idx_path = self.local_dir / "gloss2idx.pkl"
        self.senses_path = self.local_dir / "senses.pkl"
        self.idx2gloss_path = self.local_dir / "idx2gloss.pkl"
        self.embeddings_path = self.local_dir / "embeddings.pkl"
        self.triplet_path = self.local_dir / "triples.duckdb"
        self.attrs_path = self.local_dir / "attrs.pkl"
        self.forms_lexicon_path = self.local_dir / f"forms-{lang}.pkl"
        self.wiktionary_path = self.setup_wiktionary()
        self.wiktionary_size = count_jsonl(self.wiktionary_path)
        gloss2idx, idx2gloss, senses, embeddings = self.setup_senses()
        self.gloss2idx = gloss2idx
        self.idx2gloss = idx2gloss
        self.senses = senses
        self.embeddings = embeddings
        self.triples = self.setup_triple_store()
        self.lexicon = self.setup_lexicon()
        self.attrs = self.setup_attrs()
        self.api.upload_folder(
            folder_path=self.local_dir,
            repo_id=self.repo_id,
            repo_type=self.repo_type,
            ignore_patterns=["*.jsonl", "*.jsonl.gz"],
        )

    def setup_wiktionary(self) -> Path:
        logger.info("Downloading wiktionary data.")
        if not self.extracted_path.exists():
            response = requests.get(self.wiktionary_url, stream=True)
            response.raise_for_status()  # Raise an exception for HTTP errors
            total_size = response.headers.get("content-length")

            # remove the file if it exists
            if self.download_path.exists():
                os.remove(self.download_path)

            # Download the file in chunks
            with open(self.download_path, "wb") as f:
                if total_size is None:  # no content length header
                    f.write(response.content)
                else:
                    total_size = int(total_size)
                    progress_bar = tqdm.tqdm(
                        total=total_size,
                        unit="B",
                        unit_scale=True,
                        desc=f"Downloading {os.path.basename(self.download_path)}",
                        ascii=True,
                    )
                    for chunk in response.iter_content(chunk_size=4096):
                        if chunk:  # filter out keep-alive chunks
                            f.write(chunk)
                            progress_bar.update(len(chunk))
                    progress_bar.close()
            logger.info("Download complete.")

            # Extract the gzip file
            logger.info("Extracting wiktionary data.")
            with gzip.open(self.download_path, "rb") as f_in:
                with open(self.extracted_path, "wb") as f_out:
                    # Create a progress bar for extraction (can't know size in advance for gzip)
                    shutil.copyfileobj(f_in, f_out)

            logger.info("Extraction complete.")

            # Optionally, remove the compressed file after extraction
            os.remove(self.download_path)
        else:
            logger.info("Wiktionary data already exists.")
        return self.extracted_path

    def setup_senses(self):
        if self.gloss2idx_path.exists() and self.senses_path.exists():
            logger.info("Gloss2idx and senses already exist.")
            with open(self.gloss2idx_path, "rb") as f:
                gloss2idx = pickle.load(f)
            with open(self.senses_path, "rb") as f:
                senses = pickle.load(f)
        else:
            senses: dict[str, set[int]] = {}
            gloss2idx: dict[str, int] = {}

            def add_to_index(
                word: Union[Word, str], gloss: str, add_to_senses: bool = False
            ):
                if not isinstance(word, str):
                    word = word.word
                gloss_index = gloss2idx.setdefault(gloss, len(gloss2idx))
                if add_to_senses:
                    senses[word].add(gloss_index)

            progress_bar = tqdm.tqdm(
                total=self.wiktionary_size,
                desc="Extracting glosses",
                leave=True,
            )

            for word in parse_jsonl(self.wiktionary_path):
                # skip non-English words
                if word.lang_code and word.lang_code != "en":
                    progress_bar.update(1)
                    continue

                for word_sense in word.senses or []:
                    if word.word not in senses:
                        senses[word.word] = set()
                    word_sense_gloss = None
                    if word_sense.glosses:
                        word_sense_gloss = " ".join(
                            word_sense.glosses or []
                        ).strip()
                    # DO NOT USE raw_glosses since those are either obsolete words
                    #   or not properly defined terms
                    if word_sense_gloss is None:
                        continue
                    add_to_index(word, word_sense_gloss, add_to_senses=True)

                for predicate in self.predicates:
                    # word level relations
                    relations = getattr(word, f"{predicate}s") or []
                    for relation in relations:
                        if not relation.word:  # object
                            continue
                        relation_sense = getattr(relation, "sense", None)
                        if relation_sense is None:
                            continue
                        add_to_index(word, relation_sense)
                    # sense level relations
                    for word_sense in word.senses or []:
                        relations = getattr(word_sense, f"{predicate}s") or []
                        for relation in relations:
                            if not relation.word:  # object
                                continue
                            sense_gloss = None
                            # `relation_sense` this is likely None
                            relation_sense = getattr(relation, "sense", None)
                            if word_sense.glosses:
                                sense_gloss = " ".join(
                                    word_sense.glosses or []
                                ).strip()
                            elif relation_sense:
                                sense_gloss = relation_sense
                            # don't use `word_sense.raw_glosses` because they are not good ones
                            if sense_gloss is None:
                                continue
                            add_to_index(word, sense_gloss)

                progress_bar.update(1)

            progress_bar.close()

            with open(self.gloss2idx_path, "wb") as f:
                pickle.dump(gloss2idx, f)

            with open(self.senses_path, "wb") as f:
                pickle.dump(senses, f)

            logger.info("Gloss2idx and senses created.")

        if self.idx2gloss_path.exists() and self.embeddings_path.exists():
            logger.info("Idx2gloss and embeddings already exist.")
            with open(self.idx2gloss_path, "rb") as f:
                idx2gloss = pickle.load(f)
            with open(self.embeddings_path, "rb") as f:
                embeddings = pickle.load(f)
        else:
            logger.info("Creating idx2gloss and embeddings.")

            idx2gloss: dict[int, str] = {}
            embeddings: dict[int, np.ndarray] = {}

            model = AutoEmbedding.from_config({
                "loader": "transformers",
                "name": "sentence-transformers/all-MiniLM-L6-v2",
            })

            progress_bar = tqdm.tqdm(
                total=len(gloss2idx),
                desc="Creating idx2gloss and embeddings",
                leave=True,
            )

            buffer = ([], [])

            for gloss, index in gloss2idx.items():
                buffer[0].append(index)
                buffer[1].append(gloss)

                idx2gloss[index] = gloss

                if len(buffer[0]) >= 512:
                    ex = model.encode(buffer[1], batch_size=256)
                    embeddings.update(dict(zip(buffer[0], ex)))
                    buffer = ([], [])

                progress_bar.update(1)

            if buffer[0]:
                ex = model.encode(buffer[1], batch_size=256)
                embeddings.update(dict(zip(buffer[0], ex)))
                buffer = ([], [])

            progress_bar.close()

            with open(self.idx2gloss_path, "wb") as f:
                pickle.dump(idx2gloss, f)

            with open(self.embeddings_path, "wb") as f:
                pickle.dump(embeddings, f)

            logger.info("Idx2gloss and embeddings created.")

        return gloss2idx, idx2gloss, senses, embeddings

    def closest_sense(self, word: str, sense_gloss: Union[str, None]):
        if word not in self.senses:
            return None
        word_sense_indexes: list[int] = list(self.senses[word])
        if not word_sense_indexes:
            return
        sense_index = (
            self.gloss2idx[sense_gloss] if sense_gloss is not None else None
        )
        if sense_index in word_sense_indexes:
            return sense_index
        sense_embeddings = np.array([
            self.embeddings[index] for index in word_sense_indexes
        ])
        if sense_index is None:
            sense_embedding = np.array([np.mean(sense_embeddings, axis=0)])
        elif isinstance(sense_index, int):
            sense_embedding = self.embeddings[sense_index]
        else:
            raise ValueError("sense_index must be int or None")
        distances = np.linalg.norm(sense_embeddings - sense_embedding, axis=1)
        return word_sense_indexes[np.argmin(distances)]

    def get_relations(self, word: Word):
        for predicate in self.predicates:
            relations = getattr(word, f"{predicate}s") or []
            for relation in relations:
                if not relation.word:  # object
                    continue
                relation_sense = getattr(relation, "sense", None)
                if relation_sense is None:
                    continue
                subject = (
                    word.word,
                    self.closest_sense(word.word, relation_sense),
                )
                object_ = (relation.word, None)
                yield normalize(subject, predicate, object_)
            for word_sense in word.senses or []:
                relations = getattr(word_sense, f"{predicate}s") or []
                for relation in relations:
                    if not relation.word:  # object
                        continue
                    sense_gloss = None
                    # `relation_sense` this is likely None
                    relation_sense = getattr(relation, "sense", None)
                    if word_sense.glosses:
                        sense_gloss = " ".join(
                            word_sense.glosses or []
                        ).strip()
                    elif relation_sense:
                        sense_gloss = relation_sense
                    # don't use `word_sense.raw_glosses` because they are not good ones
                    subject = (
                        word.word,
                        self.closest_sense(word.word, sense_gloss),
                    )
                    object_ = (relation.word, None)
                    yield normalize(subject, predicate, object_)

    def setup_triple_store(self):
        if not self.triplet_path.exists():
            logger.info("Creating triplet store.")
            triples = TripleStore(self.triplet_path, read_only=False)

            progress_bar = tqdm.tqdm(
                total=self.wiktionary_size,
                desc="Processing Wiktionary",
                leave=True,
            )

            for word in parse_jsonl(self.wiktionary_path):
                # skip non-English words
                if word.lang_code and word.lang_code != "en":
                    progress_bar.update(1)
                    continue

                for relation in self.get_relations(word):
                    subject, predicate, object_ = relation
                    subj_word, subj_sense = subject
                    if subj_sense is None:
                        continue
                    triple = ((subj_word, subj_sense), predicate, object_)
                    triples.insert(triple)

                progress_bar.update(1)

            progress_bar.close()

            triples.close()

            logger.info("Triplet store created.")
        return TripleStore(self.triplet_path, read_only=True)

    def setup_lexicon(self) -> Lexicon[set[str]]:
        """Returns the lexicon of forms."""
        try:
            return Lexicon.load(self.forms_lexicon_path)
        except FileNotFoundError:
            pass
        lexicon = Lexicon()
        progress_bar = tqdm.tqdm(
            total=self.wiktionary_size, desc="Adding forms", unit="word"
        )
        for word in parse_jsonl(self.wiktionary_path):
            if any([
                self.lang_code == (word.lang_code or "").lower(),
                self.lang_code == (word.lang or "").lower(),
            ]):
                add_forms(word, lexicon=lexicon)
            progress_bar.update(1)
        progress_bar.close()
        lexicon.dump(self.forms_lexicon_path)
        return lexicon

    def setup_attrs(self) -> dict[str, set[int]]:
        if self.attrs_path.exists():
            with open(self.attrs_path, "rb") as f:
                attrs = pickle.load(f)
        else:
            attrs = {}

        attr_key = "hate_related"
        if attr_key not in attrs:
            hate_related_senses = set()
            batch = []

            # Download from the 🤗 Hub
            model = SetFitModel.from_pretrained(
                "ysenarath/all-MiniLM-L6-v2-hateful-definitions-full-bin-v1"
            )

            progress_bar = tqdm.tqdm(
                total=len(self.idx2gloss),
                desc="Detecting hate-related definitions",
                leave=True,
            )

            for idx, gloss in self.idx2gloss.items():
                batch.append((idx, gloss))
                if len(batch) < 1000:
                    continue
                keys, values = zip(*batch)
                labels = model(values)
                hate_related_senses.update(
                    (
                        idx
                        for idx, label in zip(keys, labels)
                        if label != "normal"
                    )
                )
                progress_bar.update(len(batch))
                batch = []

            if batch:
                keys, values = zip(*batch)
                labels = model(values)
                hate_related_senses.update(
                    (
                        idx
                        for idx, label in zip(keys, labels)
                        if label != "normal"
                    )
                )
                progress_bar.update(len(batch))

            progress_bar.close()

            attrs[attr_key] = hate_related_senses

        with open(self.attrs_path, "wb") as f:
            pickle.dump(attrs, f)

        return attrs
