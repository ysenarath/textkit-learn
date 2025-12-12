from __future__ import annotations

import gzip
import os
import pickle
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import requests
from diskcache import Cache
from huggingface_hub import HfApi
from tqdm import auto as tqdm

from tklearn import config, logging
from tklearn.embeddings.base import AutoEmbedding
from tklearn.kb.base import ArtifactStore, ArtifactStoreConfig
from tklearn.kb.lexicon import Lexicon
from tklearn.kb.triple_store import TripleStore
from tklearn.kb.wiktionary.helpers import JSONDisk
from tklearn.kb.wiktionary.models import Sense, Word, parse_jsonl

logger = logging.get_logger(__name__)
G = int | None
WS = tuple[str, G]

WIKTIONARY_URL = "https://kaikki.org/dictionary/raw-wiktextract-data.jsonl.gz"


def count_lines_fast(path):
    return int(subprocess.check_output(f"wc -l {path}", shell=True).split()[0])


def setup_wiktionary_words(
    extracted_path: Path, wiktionary_path: Path, language: str = "en"
) -> bool:
    words = []
    data_iter = parse_jsonl(extracted_path)
    num_lines = count_lines_fast(extracted_path)
    if language in {"en", "english"}:
        languages = {"en", "english"}
    else:
        raise NotImplementedError(f"Language {language} not supported.")
    for wd in tqdm.tqdm(data_iter, total=num_lines):
        if wd.lang is not None and (
            wd.lang.lower() not in languages
            or wd.lang_code.lower() not in languages
        ):
            continue
        words.append(wd)
    with open(wiktionary_path, "wb") as f:
        pickle.dump(words, f)
    return True


def download_wiktionary(wiktionary_url: str, output_path: Path):
    logger.info(f"Downloading from {wiktionary_url}")
    response = requests.get(wiktionary_url, stream=True)
    response.raise_for_status()  # Raise an exception for HTTP errors
    total_size = response.headers.get("content-length")
    # remove the file if it exists
    if output_path.exists():
        os.remove(output_path)
    # Download the file in chunks
    with open(output_path, "wb") as f:
        if total_size is None:  # no content length header
            f.write(response.content)
        else:
            total_size = int(total_size)
            progress_bar = tqdm.tqdm(
                total=total_size,
                unit="B",
                unit_scale=True,
                desc=f"Downloading {os.path.basename(output_path)}",
                ascii=True,
            )
            for chunk in response.iter_content(chunk_size=4096):
                if chunk:  # filter out keep-alive chunks
                    f.write(chunk)
                    progress_bar.update(len(chunk))
            progress_bar.close()


def setup_wiktionary(
    wiktionary_url: str,
    temp_download_path: Path,
    temp_extracted_path: Path,
    wiktionary_path: Path,
    language: str = "en",
    remove_downloaded: bool = True,
) -> Path:
    logger.info("Downloading wiktionary data.")
    if wiktionary_path is None or not wiktionary_path.exists():
        logger.info("Wiktionary data not found.")
        if not temp_extracted_path.exists():
            # Download the gzip file
            download_wiktionary(
                wiktionary_url=wiktionary_url,
                output_path=temp_download_path,
            )
            logger.info("Download complete.")
            # Extract the gzip file
            logger.info("Extracting wiktionary data.")
            with gzip.open(temp_download_path, "rb") as f_in:
                with open(temp_extracted_path, "wb") as f_out:
                    # Create a progress bar for extraction (can't know size in advance for gzip)
                    shutil.copyfileobj(f_in, f_out)
            logger.info("Extraction complete.")
            # remove the compressed file after extraction
            os.remove(temp_download_path)
        # Cache English words
        if wiktionary_path is not None:
            setup_wiktionary_words(
                extracted_path=temp_extracted_path,
                wiktionary_path=wiktionary_path,
                language=language,
            )
        # remove the extracted jsonl file after caching
        if remove_downloaded:
            os.remove(temp_extracted_path)
    else:
        logger.info("Wiktionary data already exists.")
    return temp_extracted_path


class WikitionaryProcessor:
    def __init__(
        self, words: list[Word], predicates: list[str], cache_dir: Path
    ):
        self.words = words
        self.predicates = predicates
        self.cache_dir = cache_dir

    def get_or_set_sense_id(self, definition: str | None) -> int | None:
        if definition is None:
            return None
        if definition not in self.gloss2idx:
            self.gloss2idx[definition] = len(self.gloss2idx)
        return self.gloss2idx[definition]

    def process_sense(self, word: Word, sense: Sense):
        # Determine sense ID based on glosses
        definition = " ".join(sense.glosses or []).strip() or None
        if not definition:
            definition = " ".join(sense.raw_glosses or []).strip() or None
        sense_id = None
        if definition:
            sense_id = self.get_or_set_sense_id(definition)
            self.sense2words[sense_id].add(word.word)
        # Add form-of relations
        for form_of in sense.form_of or []:
            self.form2senses[word.word].add((form_of.word, sense_id))
        # Add other relations
        for predicate in self.predicates:
            objects = getattr(sense, predicate + "s") or []
            for obj in objects:
                obj_sense_id = None
                if obj.sense:
                    assert isinstance(obj.sense, str), (
                        "expected str, got {}".format(type(obj.sense).__name__)
                    )
                    obj_sense_id = self.get_or_set_sense_id(obj.sense)
                    self.sense2words[obj_sense_id].add(word.word)
                triple = (
                    (word.word, sense_id),
                    predicate,
                    (obj.word, obj_sense_id),
                )
                self.triple_set.add(triple)

    def process_word(self, word: Word):
        self.form2senses[word.word].add((word.word, None))
        for form in word.forms or []:
            word_form: str = form.form
            self.form2senses[word_form].add((word.word, None))
        for predicate in self.predicates:
            objects = getattr(word, predicate + "s") or []
            for obj in objects:
                obj_sense_id = None
                if obj.sense:
                    assert isinstance(obj.sense, str), (
                        "expected str, got {}".format(type(obj.sense).__name__)
                    )
                    obj_sense_id = self.get_or_set_sense_id(obj.sense)
                    self.sense2words[obj_sense_id].add(word.word)
                triple = (
                    (word.word, None),
                    predicate,
                    (obj.word, obj_sense_id),
                )
                self.triple_set.add(triple)
        for sense in word.senses or []:
            self.process_sense(word, sense)

    def process_words(self):
        self.triple_set = set()
        self.sense2words: defaultdict[G, set[str]] = defaultdict(set)
        self.form2senses: defaultdict[str, set[WS]] = defaultdict(set)
        self.gloss2idx: dict[str, int] = {}
        try:
            with open(self.cache_dir / "processed-words.pkl", "rb") as f:
                cache_data = pickle.load(f)
            self.triple_set = cache_data["triple_set"]
            self.sense2words = cache_data["sense2words"]
            self.form2senses = cache_data["form2senses"]
            self.gloss2idx = cache_data["gloss2idx"]
        except FileNotFoundError:
            for word in tqdm.tqdm(self.words):
                self.process_word(word)
            # save intermediate cache
            cache_data = {
                "triple_set": self.triple_set,
                "sense2words": self.sense2words,
                "form2senses": self.form2senses,
                "gloss2idx": self.gloss2idx,
            }
            with open(self.cache_dir / "processed-words.pkl", "wb") as f:
                pickle.dump(cache_data, f)

    def update_embeddings(
        self, buffer: tuple[list[int], list[str]], ex: np.ndarray
    ):
        self.embeddings.update(dict(zip(buffer[0], ex)))
        for k, v in zip(buffer[1], ex):
            self.embedding_cache[k] = v

    def embed_glosses(self) -> dict[int, np.ndarray]:
        logger.info("Building gloss embeddings.")
        model = None
        desc = "Embedding Definitions"
        progress_bar = tqdm.tqdm(total=len(self.gloss2idx), desc=desc)
        buffer = ([], [])
        for gloss, index in self.gloss2idx.items():
            if gloss in self.embedding_cache:
                self.embeddings[index] = self.embedding_cache[gloss]
                progress_bar.update(1)
                continue
            if model is None:
                model = AutoEmbedding({
                    "loader": "transformers",
                    "name": "sentence-transformers/all-MiniLM-L6-v2",
                })
            buffer[0].append(index)
            buffer[1].append(gloss)
            if len(buffer[0]) >= 10_000:
                ex = model.encode(buffer[1], batch_size=256)
                self.update_embeddings(buffer, ex)
                buffer = ([], [])
            progress_bar.update(1)
        if buffer[0]:
            ex = model.encode(buffer[1], batch_size=256)
            self.update_embeddings(buffer, ex)
            buffer = ([], [])
        progress_bar.close()
        logger.info("Completed building gloss embeddings.")

    def build(self) -> dict[str, Any]:
        self.process_words()

        ts = TripleStore()

        def add_triple(triple: tuple[WS, str, WS]):
            subject, predicate, object_ = triple
            if subject[1] is None and object_[1] is None:
                return
            if predicate == "hyponym":
                rev_triple = (object_, "hypernym", subject)
                ts.insert(rev_triple)
            else:
                ts.insert(triple)

        for triple in self.triple_set:
            add_triple(triple)

        lexicon: Lexicon[set[tuple[str, int]]]
        try:
            lexicon = Lexicon.load(self.cache_dir / "lexicon.pkl")
        except FileNotFoundError:
            lexicon = Lexicon()
            for form, words_set in self.form2senses.items():
                if form is None:
                    continue
                form = form.strip()
                if not form:
                    continue
                if form in lexicon:
                    lexicon[form].update({
                        word for word in words_set if word[0] is not None
                    })
                else:
                    lexicon[form] = {
                        word for word in words_set if word[0] is not None
                    }
            lexicon.dump(self.cache_dir / "lexicon.pkl")

        try:
            with open(self.cache_dir / "senses.pkl", "rb") as f:
                senses = pickle.load(f)
        except FileNotFoundError:
            senses = defaultdict(set)
            for sense_id, words in self.sense2words.items():
                for word in words:
                    senses[word].add(sense_id)
            with open(self.cache_dir / "senses.pkl", "wb") as f:
                pickle.dump(senses, f)

        pth = self.cache_dir / "embeddings"
        self.embedding_cache = Cache(pth, disk=JSONDisk)
        self.embeddings = {}
        self.embed_glosses()
        self.embedding_cache.close()

        return {
            "triples": ts,
            "lexicon": lexicon,
            "gloss2idx": self.gloss2idx,
            "idx2gloss": {idx: gloss for gloss, idx in self.gloss2idx.items()},
            "senses": senses,
            "embeddings": self.embeddings,
            "attrs": {},
        }


class WiktionaryArtifactStoreConfig(ArtifactStoreConfig):
    name: ClassVar[str] = "wiktionary"
    username_or_org: str = "textkit-learn"
    private: bool = True
    version: str = "v1.0"
    repo_type: str = "dataset"
    language: str = "en"
    verbose: bool = False

    @property
    def repo_name(self) -> str:
        return f"{self.name}-{self.version}"

    @property
    def repo_id(self) -> str:
        return f"{self.username_or_org}/{self.repo_name}"


class WiktionaryArtifactStore(ArtifactStore):
    config: WiktionaryArtifactStoreConfig

    # form (str) -> set of words (set[str])
    lexicon: Lexicon[set[str]]
    # subject (str, int) -> predicate (str) -> object (str, int)
    triples: TripleStore
    # gloss (str) -> index (int)
    gloss2idx: dict[str, int]
    # index (int) -> gloss (str)
    idx2gloss: dict[int, str]
    # word (str) -> set of senses (set[int])
    senses: dict[str, set[int]]
    # sense (int) -> embedding (np.ndarray)
    embeddings: dict[int, np.ndarray]
    # attribute (str) -> set of senses (set[int])
    attrs: dict[str, set[int]]

    def __post_init__(self):
        # local_dir is the directory where the repo will be downloaded
        self.local_dir = Path(config.assets_dir) / self.config.repo_name
        self.local_dir.mkdir(parents=True, exist_ok=True)
        # must be compatible with wiktextract library (https://kaikki.org)
        # Hugging Face Hub API
        self.api = HfApi()
        # create the repo if it does not exist
        self.api.create_repo(
            repo_id=self.config.repo_id,
            private=self.config.private,
            repo_type=self.config.repo_type,
            exist_ok=True,
        )
        # download the repo to the local directory
        self.api.snapshot_download(
            repo_id=self.config.repo_id,
            repo_type=self.config.repo_type,
            local_dir=self.local_dir,
        )
        language = self.config.language
        self.wiktionary_path = self.local_dir / "words.pkl"
        setup_wiktionary(
            wiktionary_url=WIKTIONARY_URL,
            temp_download_path=self.local_dir / "wiktionary.jsonl.gz",
            temp_extracted_path=self.local_dir / "wiktionary.jsonl",
            wiktionary_path=self.wiktionary_path,
            language=language,
        )
        processor = self.get_processor()
        for key, value in processor.build().items():
            setattr(self, key, value)
        self.api.upload_folder(
            folder_path=self.local_dir,
            repo_id=self.config.repo_id,
            repo_type=self.config.repo_type,
            ignore_patterns=["*.jsonl", "*.jsonl.gz"],
        )
        if self.config.verbose:
            logger.setLevel(logging.WARNING)
        else:
            logger.setLevel(logging.INFO)

    def get_processor(self):
        with open(self.wiktionary_path, "rb") as f:
            words: list[Word] = pickle.load(f)
        return WikitionaryProcessor(
            words=words,
            predicates=["synonym", "antonym", "hypernym", "hyponym"],
            cache_dir=self.local_dir,
        )
