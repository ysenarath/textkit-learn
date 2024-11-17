from __future__ import annotations

import operator
import string
from abc import ABC
from pathlib import Path
from typing import Iterable, List, cast

import faiss
import nltk
import numpy as np
import pandas as pd
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import Lemma, Synset
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
from tklearn.config import config
from tqdm.auto import tqdm

# Download WordNet if not already downloaded
nltk.download("wordnet")


def get_wordnet_pos(pos_tag: str):
    """Convert Penn Treebank POS tags to WordNet POS tags"""
    if pos_tag.startswith("J"):
        return wn.ADJ
    elif pos_tag.startswith("V"):
        return wn.VERB
    elif pos_tag.startswith("N"):
        return wn.NOUN
    elif pos_tag.startswith("R"):
        return wn.ADV
    else:
        return None


def get_context_window(
    tokens: list[str], target_word: str, window_size: int = -1
) -> List[str]:
    """Get words around the target word within a fixed window size"""
    tokens = [token.lower() for token in tokens]
    if window_size < 0:
        return tokens
    try:
        target_idx = tokens.index(target_word.lower())
        start = max(0, target_idx - window_size)
        end = min(len(tokens), target_idx + window_size + 1)
        return tokens[start:end]
    except ValueError:
        return tokens


def get_related_words(synset: Synset, depth: int = 2) -> List[str]:
    related: List[str] = []
    # Add words from definition
    related.extend(word_tokenize(synset.definition()))
    # Add example sentences
    for example in synset.examples():
        related.extend(word_tokenize(example))
    # Add lemma names
    lemmas_: List[Lemma] = synset.lemmas()
    related.extend(sum([word_tokenize(lemma.name()) for lemma in lemmas_], []))
    # Add hypernyms and hyponyms up to specified depth
    if depth > 0:
        for hypernym in synset.hypernyms():
            related.extend(get_related_words(hypernym, depth - 1))
        for hyponym in synset.hyponyms():
            related.extend(get_related_words(hyponym, depth - 1))
    # Clean and normalize
    related = [
        word.lower()
        for word in related
        if word.lower() not in stopwords.words("english")
        and word not in string.punctuation
    ]
    return related


def compute_score_by_overlap(
    context_words: Iterable[str], sense_words: Iterable[str]
) -> int:
    return len(set(context_words).intersection(sense_words))


def lesk_wsd(args: dict | pd.Series) -> List[dict]:
    if isinstance(args, pd.Series):
        args = args.to_dict()
    tokens: List[str] = args["tokens"]
    target_word: str = args["token"]
    pos: str | None = args.get("pos_tag")
    # Get context window around target word
    context_words = get_context_window(tokens, target_word)
    # If POS is not provided, try to determine it
    if pos is None:
        tagged = pos_tag([target_word])[0][1]
        pos = get_wordnet_pos(tagged)
    # Get all possible synsets for the target word
    candidate_synsets: List[Synset]
    if pos:
        candidate_synsets = wn.synsets(target_word, pos)
    else:
        candidate_synsets = wn.synsets(target_word)
    if not candidate_synsets:
        return []
    # Find the synset with the highest overlap with context
    candicates = []
    for sense in candidate_synsets:
        # Get related words for this sense
        sense_words = get_related_words(sense)
        # Compute overlap between context and sense words
        score = compute_score_by_overlap(context_words, sense_words)
        sense_name = sense.name()
        sense_def = sense.definition()
        candicates.append({
            "synset": sense_name,
            "sense_def": sense_def,
            "score": score,
        })
    return candicates


def get_wordnet_examples() -> pd.DataFrame:
    """Retrieve all examples from WordNet with their associated words/phrases"""
    examples = []
    # Iterate through all synsets in WordNet
    all_sunsets: List[Synset] = list(wn.all_synsets())
    for synset in list(all_sunsets):
        # Get examples for the synset
        for example in synset.examples():
            # Get all lemma names (words) associated with this synset
            for lemma in synset.lemmas():
                lemma = cast(Lemma, lemma)
                lemma_str = lemma.name()
                # check lemma in example
                if lemma_str in example:
                    examples.append({
                        "lemma": lemma_str,
                        "example": example,
                        "pos": synset.pos(),
                        "synset": synset.name(),
                        "definition": synset.definition(),
                    })
    return pd.DataFrame(examples)


def create_prompt(
    row: dict = None, /, *, lemma: str = None, example: str = None
) -> str:
    """Create a prompt in the specified format"""
    lemma = row["lemma"] if lemma is None else lemma
    example = row["example"] if example is None else example
    return f'Given the word "{lemma}", here is an example of its usage: "{example}"'


def create_prompts(df: pd.DataFrame) -> List[str]:
    """Create prompts in the specified format"""
    return df.apply(create_prompt, axis=1).tolist()


def iter_generate_embeddings(
    texts: List[str],
    model: SentenceTransformer,
    batch_size: int = 32,
) -> Iterable[np.ndarray]:
    """Generate embeddings for a list of texts using sentence-transformers"""
    # Generate embeddings
    # Process in batches to manage memory
    kwargs = {"desc": "Generating embeddings"}
    for i in tqdm(range(0, len(texts), batch_size), **kwargs):
        batch = texts[i : i + batch_size]
        batch_embeddings = model.encode(batch, convert_to_tensor=True)
        yield batch_embeddings.cpu().numpy()


def generate_embeddings(
    texts: List[str],
    model: SentenceTransformer,
    batch_size: int = 32,
) -> np.ndarray:
    """Generate embeddings for a list of texts using sentence-transformers"""
    # Generate embeddings
    embeddings = []
    for batch_embeddings in iter_generate_embeddings(
        texts, model=model, batch_size=batch_size
    ):
        embeddings.append(batch_embeddings)
    # Concatenate all embeddings
    return np.concatenate(embeddings, axis=0)


def create_vector_db(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    # get dimensionality of embeddings
    d = embeddings.shape[1]
    # create FAISS index
    # using L2 (Euclidean) distance for similarity search
    index = faiss.IndexFlatL2(d)
    # add vectors to the index
    index.add(embeddings.astype(np.float32))  # FAISS requires float32
    # return index
    return index


class WordSenseDisambiguator(ABC):
    def __init__(self, *, cache_dir: str | None = None):
        self.stopwords = set(stopwords.words("english"))
        self.tokenize = word_tokenize
        self.pos_tag = pos_tag
        self.batch_size = 32
        self.embedding_model_name_or_path = (
            "sentence-transformers/all-mpnet-base-v2"
        )
        self.k = 5
        self.cache_dir = (
            Path(cache_dir) if cache_dir else Path(config.cache_dir) / "wsd"
        )
        self.__post_init__()

    def __post_init__(self):
        # cache_dir
        model = SentenceTransformer(self.embedding_model_name_or_path)
        # Get examples and create prompts
        df = get_wordnet_examples()
        vector_index_path = self.cache_dir / "vector_index-v0.2.faiss"
        # create parent folder if not exists
        vector_index_path.parent.mkdir(parents=True, exist_ok=True)
        if vector_index_path.exists():
            self.vector_index = faiss.read_index(str(vector_index_path))
        else:
            prompts = create_prompts(df)
            # Generate embeddings
            embeddings = generate_embeddings(
                prompts, model, batch_size=self.batch_size
            )
            # Create FAISS index
            self.vector_index = create_vector_db(embeddings)
            faiss.write_index(self.vector_index, str(vector_index_path))
        self.embedding_model = model
        self.wordnet_examples_df = df

    def find_candidates(self, df: pd.DataFrame) -> List[List[str]]:
        candicates = []
        prompts = df.apply(
            lambda row: create_prompt(lemma=row["token"], example=row["text"]),
            axis=1,
        ).tolist()
        for batch_embeddings in iter_generate_embeddings(
            prompts,
            self.embedding_model,
            batch_size=self.batch_size,
        ):
            # D[i, j] contains the distance from the i-th
            #   query vector to its j-th nearest neighbor.
            # I[i, j] contains the id of the j-th nearest
            #   neighbor of the i-th query vector.
            D, I = self.vector_index.search(batch_embeddings, self.k)
            for i in range(len(batch_embeddings)):
                item_candicates = []
                for j in range(self.k):
                    distance = D[i, j]
                    neighbor_id = I[i, j]
                    neighbor_row = self.wordnet_examples_df.iloc[neighbor_id]
                    synset = neighbor_row["synset"]
                    definition = neighbor_row["definition"]
                    item_candicates.append({
                        "distance": distance,
                        "synset": synset,
                        "definition": definition,
                    })
                candicates.append(item_candicates)
        return candicates

    def disambiguate(self, text: str | List[str] | pd.Series) -> List[str]:
        if isinstance(text, str):
            return self.disambiguate([text])
        df = pd.DataFrame(text, columns=["text"])
        df.reset_index(drop=False, names="index", inplace=True)
        tokens = df["text"].apply(self.tokenize)
        df = df.assign(
            tokens=tokens,
            token=tokens.apply(
                lambda x: [(i, *t) for i, t in enumerate(self.pos_tag(x))]
            ),
        ).explode("token", ignore_index=True)
        df = df.assign(
            token_index=df["token"].apply(operator.itemgetter(0)),
            token=df["token"].apply(operator.itemgetter(1)),
            pos_tag=df["token"]
            .apply(operator.itemgetter(2))
            .apply(get_wordnet_pos),
        )
        type_1_cand = self.find_candidates(df)
        type_2_cand = df.apply(lesk_wsd, axis=1)
        candicates = []
        for i in range(len(type_1_cand)):
            cand_1 = {item["synset"]: item for item in type_1_cand[i]}
            cand_2 = {item["synset"]: item for item in type_2_cand[i]}
            item_candicates = []
            for key in set(cand_1.keys()).union(cand_2.keys()):
                item_candicates.append({
                    "synset": key,
                    "distance": cand_1.get(key, {}).get("distance", None),
                    "score": cand_2.get(key, {}).get("score", None),
                    "definition": cand_1.get(key, cand_2.get(key, {})).get(
                        "definition", None
                    ),
                })
            candicates.append(item_candicates)
        df = (
            df.assign(candicates=candicates)
            .explode("candicates", ignore_index=True)
            .reset_index(drop=True)
        )
        # .rename(columns=lambda x: f"candidates.{x}"),
        df = pd.concat(
            [
                df,
                df["candicates"].apply(pd.Series),
            ],
            axis=1,
        )
        return df
