from __future__ import annotations

import gzip
import os
import pickle
import re
import shutil
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import requests
from datasets import Dataset
from huggingface_hub import HfApi
from tqdm import auto as tqdm

from tklearn import config, logging
from tklearn.embeddings.base import AutoEmbedding, Embedding
from tklearn.exceptions import UnexpectedValueError
from tklearn.kb.base import ArtifactStore, ArtifactStoreConfig
from tklearn.kb.lexicon import Lexicon
from tklearn.kb.triple_store_v2 import TripleStore
from tklearn.kb.wiktionary.models import Sense, Word, parse_jsonl

logger = logging.get_logger(__name__)
G = int | None
WS = tuple[str, G]

WIKTIONARY_URL = "https://kaikki.org/dictionary/raw-wiktextract-data.jsonl.gz"
SUPPORTED_PREDICATES = {
    "synonym",
    "antonym",
    "hypernym",
    "hyponym",
    "category",
}


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
    desc = "Caching Wiktionary"
    for wd in tqdm.tqdm(data_iter, total=num_lines, desc=desc):
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


def load_from_cache(data: list[dict[str, Any]]):
    for item in data:
        yield item


def batch_embedding_func(
    batch: dict[str, list[Any]], *, encoder: Embedding
) -> dict[str, list[np.ndarray]]:
    try:
        embeddings = encoder.encode_document(batch["gloss"])
    except NotImplementedError:
        embeddings = encoder.encode(batch["gloss"])
    return {"embedding": embeddings}


def compute_gloss_embeddings(
    gloss2idx: dict[str, int], cache_file_name: str | Path
) -> dict[int, np.ndarray]:
    cache_file_name = Path(cache_file_name)
    if not cache_file_name.exists():
        ds = Dataset.from_generator(
            load_from_cache,
            gen_kwargs={
                "data": [{"gloss": gloss} for gloss in gloss2idx.keys()]
            },
        )
        ds = ds.map(
            batch_embedding_func,
            batched=True,
            batch_size=10_000,
            num_proc=1,
            load_from_cache_file=True,
            fn_kwargs={
                "encoder": AutoEmbedding({
                    "loader": "transformers",
                    "name": "sentence-transformers/all-mpnet-base-v2",
                }),
            },
        )
        ds = ds.save_to_disk(cache_file_name)
        del ds
    dataset = Dataset.load_from_disk(cache_file_name)
    dataset.set_format("numpy")
    embeddings = {}
    for item in tqdm.tqdm(dataset, desc="Computing Gloss Embeddings"):
        idx = gloss2idx[item["gloss"]]
        embeddings[idx] = item["embedding"]
    return embeddings


def format_triple(triple: tuple[WS, str, WS]):
    subject, predicate, object_ = triple
    if subject[1] is None and object_[1] is None:
        return
    if predicate == "hyponym":
        rev_triple = (object_, "hypernym", subject)
        return rev_triple
    return triple


@dataclass
class WordSense:
    word: str
    sense: str | None = None

    def __init__(self, word: str, sense: str | None = None):
        # asset word is str
        if not isinstance(word, str):
            raise UnexpectedValueError(
                got=type(word).__name__,
                expected="str",
            )
        self.word = word
        self.sense = sense


def predicate_getattr(subj: Word | Sense, predicate: str) -> list[WordSense]:
    value = None
    if predicate == "synonym":
        value = subj.synonyms
    elif predicate == "antonym":
        value = subj.antonyms
    elif predicate == "hypernym":
        value = subj.hypernyms
    elif predicate == "hyponym":
        value = subj.hyponyms
    elif predicate == "category" and subj.categories:
        value = list(map(WordSense, subj.categories))
    else:
        raise UnexpectedValueError(
            got=predicate, expected=SUPPORTED_PREDICATES
        )
    return value or []


class WikitionaryProcessor:
    ANTI_PATTERNS = [
        # antonym(s) of "{definition}" -> definition
        # (see https://en.wiktionary.org/wiki/Template:antsense)
        re.compile(r'antonym\(s\) of [“"](.*)[”"]')
    ]

    def __init__(
        self,
        wiktionary_path: Path,
        cache_dir: Path,
        predicates: set[str] | None = None,
    ):
        self.wiktionary_path = wiktionary_path
        if predicates is None:
            predicates = SUPPORTED_PREDICATES
        self.predicates = set(predicates)
        self.cache_dir = cache_dir

    def get_or_set_sense_id(self, definition: str | None) -> int | None:
        if definition is None:
            return None
        if not isinstance(definition, str):
            raise UnexpectedValueError(
                got=type(definition).__name__,
                expected="str",
            )
        for pattern in self.ANTI_PATTERNS:
            m = pattern.match(definition)
            if m:
                definition = m.group(1)
                break
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
            objects = predicate_getattr(sense, predicate)
            for obj in objects:
                rel_sense_id = None
                if obj.sense:
                    assert isinstance(obj.sense, str), (
                        "expected str, got {}".format(type(obj.sense).__name__)
                    )
                    rel_sense_id = self.get_or_set_sense_id(obj.sense)
                    self.sense2words[rel_sense_id].add(word.word)
                triple = (
                    (word.word, sense_id or rel_sense_id),
                    predicate,
                    (obj.word, None),
                )
                triple = format_triple(triple)
                if triple is None:
                    continue
                self.triples.add(triple)

    def process_word(self, word: Word):
        self.form2senses[word.word].add((word.word, None))
        for form in word.forms or []:
            word_form: str = form.form
            self.form2senses[word_form].add((word.word, None))
        for predicate in self.predicates:
            objects = predicate_getattr(word, predicate)
            for obj in objects:
                rel_sense_id = None
                if obj.sense:
                    rel_sense_id = self.get_or_set_sense_id(obj.sense)
                    self.sense2words[rel_sense_id].add(word.word)
                triple = (
                    (word.word, rel_sense_id),
                    predicate,
                    (obj.word, None),
                )
                triple = format_triple(triple)
                if triple is None:
                    continue
                self.triples.add(triple)
        for sense in word.senses or []:
            self.process_sense(word, sense)

    def process_words(self):
        self.triples = TripleStore()
        self.sense2words: defaultdict[G, set[str]] = defaultdict(set)
        self.form2senses: defaultdict[str, set[WS]] = defaultdict(set)
        self.gloss2idx: dict[str, int] = {}
        try:
            logger.info("Loading processed words from cache.")
            with open(self.cache_dir / "processed-words.pkl", "rb") as f:
                cache_data = pickle.load(f)
            logger.info("Loaded processed words from cache.")
            self.triples = cache_data["triples"]
            self.sense2words = cache_data["sense2words"]
            self.form2senses = cache_data["form2senses"]
            self.gloss2idx = cache_data["gloss2idx"]
        except FileNotFoundError:
            with open(self.wiktionary_path, "rb") as f:
                words: list[Word] = pickle.load(f)
            for word in tqdm.tqdm(words, desc="Processing Words"):
                self.process_word(word)
            # save intermediate cache
            cache_data = {
                "triples": self.triples,
                "sense2words": self.sense2words,
                "form2senses": self.form2senses,
                "gloss2idx": self.gloss2idx,
            }
            with open(self.cache_dir / "processed-words.pkl", "wb") as f:
                pickle.dump(cache_data, f)

    def build(self) -> dict[str, Any]:
        self.process_words()

        lexicon: Lexicon[set[tuple[str, int]]]
        try:
            logger.info("Loading lexicon from cache.")
            lexicon = Lexicon.load(self.cache_dir / "lexicon.pkl")
            logger.info("Loaded lexicon from cache.")
        except FileNotFoundError:
            logger.info("Failed to load lexicon from cache. Building lexicon.")
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
            logger.info("Saving lexicon to cache.")
            lexicon.dump(self.cache_dir / "lexicon.pkl")
            logger.info("Completed building lexicon.")

        try:
            logger.info("Loading senses from cache.")
            with open(self.cache_dir / "senses.pkl", "rb") as f:
                senses = pickle.load(f)
            logger.info("Loaded senses from cache.")
        except FileNotFoundError:
            logger.info("Failed to load senses from cache. Building senses.")
            senses = defaultdict(set)
            for sense_id, words in self.sense2words.items():
                for word in words:
                    senses[word].add(sense_id)
            logger.info("Saving senses to cache.")
            with open(self.cache_dir / "senses.pkl", "wb") as f:
                pickle.dump(senses, f)
            logger.info("Completed building senses.")

        logger.info("Loading embeddings from cache.")
        self.embeddings = compute_gloss_embeddings(
            gloss2idx=self.gloss2idx,
            cache_file_name=self.cache_dir / "embeddings",
        )
        logger.info("Completed loading embeddings from cache.")

        return {
            "triples": self.triples,
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
        if self.config.verbose:
            logger.setLevel(logging.WARNING)
        else:
            logger.setLevel(logging.INFO)
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

    def get_processor(self):
        return WikitionaryProcessor(
            wiktionary_path=self.wiktionary_path, cache_dir=self.local_dir
        )
