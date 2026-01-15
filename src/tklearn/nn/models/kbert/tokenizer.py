from __future__ import annotations

import json
from collections import defaultdict
from itertools import tee
from pathlib import Path
from typing import Generator, Optional, TypedDict

import numpy as np
import pandas as pd
from nightjar import BaseConfig
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import RFECV, chi2, mutual_info_classif
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC, LinearSVC
from tqdm import auto as tqdm
from transformers import AutoTokenizer, PreTrainedTokenizer
from typing_extensions import Literal, Self

from tklearn.kb.base import ArtifactStoreConfig, KnowledgeBase
from tklearn.kb.models import Mention, Span, Triple
from tklearn.nn.models.kbert.helpers import Injection, inject
from tklearn.plotting.token_tree import TokenTree

__all__ = [
    "KBertTokenizerLegacyConfig",
    "KBertTokenizerLegacyConfigDict",
    "KBertTokenizer",
]


def passthrough(x):
    return x


def iter2dict(items: list[dict]) -> dict:
    batch = defaultdict(list)
    keys = set()
    it0, it1 = tee(items)
    for item in it0:
        keys.update(item.keys())
        break
    for item in it1:
        for k in keys:
            batch[k].append(item[k])
    return dict(batch)


def dict2iter(batch: dict) -> Generator[dict, None, None]:
    batch_size = len(next(iter(batch.values())))
    for i in range(batch_size):
        item = {}
        for k, v in batch.items():
            item[k] = v[i]
        yield item


def get_token_aligned_triples(item: dict) -> dict:
    if "triples" not in item or not item["triples"]:
        return []
    text = item["text"]
    offset_mapping = item["offset_mapping"]
    # build char to token index mapping
    char2token = np.zeros(len(text), dtype=np.int32) - 1
    previous_end_char = 0
    for token_idx, (start_char, end_char) in enumerate(offset_mapping):
        for char_idx in range(previous_end_char, end_char):
            char2token[char_idx] = token_idx
        previous_end_char = end_char
    # normal tokens can see each other
    triples = []
    for triple in item["triples"]:
        # get the token indexes for the mention span
        mention_start_char = triple["mention.span.start"]
        mention_end_char = triple["mention.span.end"]
        mention_start_token = char2token[mention_start_char]
        mention_end_token = char2token[mention_end_char - 1] + 1
        # get the token indexes for the triple span
        triple_start_char = triple["triple.span.start"]
        triple_end_char = triple["triple.span.end"]
        triple_start_token = char2token[triple_start_char]
        triple_end_token = char2token[triple_end_char - 1] + 1
        triples.append({
            "span.type": "token",
            # triple data
            "triple.subject": triple["triple.subject"],
            "triple.predicate": triple["triple.predicate"],
            "triple.object": triple["triple.object"],
            # triple span data
            "triple.span.start.char": triple_start_char,
            "triple.span.end.char": triple_end_char,
            "triple.span.start": triple_start_token.item(),
            "triple.span.end": triple_end_token.item(),
            "triple.text": triple["triple.text"],
            "triple.tokens": item["tokens"][
                triple_start_token:triple_end_token
            ],
            # mention span data
            "mention.span.start.char": mention_start_char,
            "mention.span.end.char": mention_end_char,
            "mention.span.start": mention_start_token.item(),
            "mention.span.end": mention_end_token.item(),
            "mention.text": triple["mention.text"],
            "mention.tokens": item["tokens"][
                mention_start_token:mention_end_token
            ],
        })
    return triples


def extract_visibility_matrix(item: dict):
    input_ids = item["input_ids"]
    seq_len = len(input_ids)
    is_normal_token = np.ones(seq_len, dtype=bool)
    visibility_matrix = np.zeros((seq_len, seq_len), dtype=np.int8)
    for triple in item["triples"]:
        if triple["span.type"] != "token":
            raise ValueError(
                "Triples must be token-aligned to extract visibility matrix."
            )
        # mention is the parent of the triple tokens
        mention_start_token = triple["mention.span.start"]
        mention_end_token = triple["mention.span.end"]
        triple_start_token = triple["triple.span.start"]
        triple_end_token = triple["triple.span.end"]
        # set visibility between mention tokens and triple tokens
        for tt in range(triple_start_token, triple_end_token):
            is_normal_token[tt] = False
            for mt in range(mention_start_token, mention_end_token):
                visibility_matrix[mt, tt] = 1
                visibility_matrix[tt, mt] = 1
            for tk in range(triple_start_token, triple_end_token):
                visibility_matrix[tt, tk] = 1
    # set visibility among normal tokens
    normal_token_indexes = np.where(is_normal_token)[0]
    for ti in normal_token_indexes:
        for tj in normal_token_indexes:
            visibility_matrix[ti, tj] = 1
    return visibility_matrix


def extract_soft_position_index(tree: TokenTree) -> np.ndarray:
    soft_position_index = np.zeros(len(tree), dtype=np.int32)
    for idx, item in enumerate(tree):
        soft_position_index[idx] = item["depth"]
    return soft_position_index


ScorerLiteral = Literal["default", "svm_score", "chi2_score", "mutual_info"]
FeaturizerLiteral = Literal["count", "tfidf"]


class KBertTokenizerLegacyConfig(BaseConfig):
    model_name_or_path: str
    predicates: Optional[list[str]] = None
    augment_top_k: Optional[int] = 2
    threshold_top_k: int = 500
    scorer: ScorerLiteral | str = "default"
    sequence_length: int = 512
    truncate: bool = True
    knowledge_base: str | dict | ArtifactStoreConfig = "wiktionary"
    featurizer: FeaturizerLiteral | str = "count"


KBertTokenizerLegacyConfig._dispatch_registry.register(
    KBertTokenizerLegacyConfig, True
)


class KBertTokenizerLegacyConfigDict(TypedDict):
    model_name_or_path: str
    predicates: list[str] | None
    augment_top_k: int | None
    threshold_top_k: int
    scorer: ScorerLiteral | str
    sequence_length: int
    truncate: bool
    knowledge_base: str | dict | ArtifactStoreConfig
    featurizer: FeaturizerLiteral | str


class KBertTokenizer:
    tokenizer: PreTrainedTokenizer
    knowledge_base: KnowledgeBase
    triple_scores: pd.DataFrame | None
    threshold: float | None

    def __init__(
        self,
        config: KBertTokenizerLegacyConfig | KBertTokenizerLegacyConfigDict,
    ):
        if isinstance(config, dict):
            config = KBertTokenizerLegacyConfig.from_dict(config)
        self.config = config
        self.__post_init__()

    def __post_init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name_or_path
        )
        self.knowledge_base = KnowledgeBase(self.config.knowledge_base)
        # this may be uploaded when .fit is called
        self.triple_scores = None
        self.threshold = None

    def save_pretrained(self, save_directory: str, **kwargs):
        sort_values_by = kwargs.pop("sort_values_by", None)
        self.tokenizer.save_pretrained(save_directory, **kwargs)
        triple_scores = self.triple_scores
        if sort_values_by and sort_values_by in triple_scores.columns:
            triple_scores = triple_scores.sort_values(
                by=sort_values_by, ascending=False
            )
        path = Path(save_directory) / "triple_scores.csv"
        triple_scores.to_csv(path)
        json_data = self.config.to_dict()
        json_data["threshold"] = self.threshold
        path = Path(save_directory) / "kb_tokenizer_config.json"
        with open(path, "w") as f:
            json.dump(json_data, f, indent=4)

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str, **kwargs
    ) -> Self:
        base_path = Path(pretrained_model_name_or_path)
        config_path = base_path / "kb_tokenizer_config.json"
        config = {}
        if config_path.exists():
            with open(config_path, "r") as f:
                config = json.load(f)
        threshold = config.pop("threshold")
        self = cls(config)
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path, **kwargs
        )
        triple_scores_path = base_path / "triple_scores.csv"
        if triple_scores_path.exists():
            self.triple_scores = pd.read_csv(triple_scores_path, index_col=0)
        self.threshold = threshold
        return self

    @property
    def pad_token_id(self) -> int:
        # bert-style padding
        return self.tokenizer.pad_token_id

    def convert_ids_to_tokens(self, ids: int | list[int]) -> str | list[str]:
        return self.tokenizer.convert_ids_to_tokens(ids)

    def extract_mentions(self, *, text: str) -> list[Mention]:
        return self.knowledge_base.extract_mentions(text)

    def filter_triples(
        self,
        *,
        relations: list[Triple],
        text: str | None = None,
        mention: Mention | None = None,
        return_all: bool = False,
    ) -> tuple[list[Triple], dict[Triple, float]]:
        # filter candidate triples based on some criteria
        if return_all:
            return relations, {}
        relation_scores = {}
        if self.triple_scores is not None and self.threshold is not None:
            filtered_relations = []
            for triple in relations:
                predicate_object = f"{triple.predicate}:{triple.object[0]}"
                try:
                    score_row = self.triple_scores.loc[predicate_object]
                except KeyError:
                    continue
                if score_row.empty:
                    continue
                if self.get_score(score_row) < self.threshold:
                    continue
                filtered_relations.append(triple)
                relation_scores[triple] = self.get_score(score_row)
            relations = filtered_relations
        return relations, relation_scores

    def extract_triples(
        self, *, text: str, return_all: bool = False
    ) -> tuple[list[tuple[str, str, str]], list[Span]]:
        top_k = None if return_all else self.config.augment_top_k
        predicates = None
        if self.config.predicates is not None:
            predicates = set(self.config.predicates)
        mentions = self.extract_mentions(text=text)
        collection = ([], [])  # triples, mention_spans
        for mention in mentions:
            curr_mention_triples = {}
            curr_mention_triple_scores = {}
            for candidate in mention.candidates:
                relations = self.knowledge_base.extract_relations(
                    candidate, predicates=predicates
                )
                candidate_triples, candidate_triple_scores = (
                    self.filter_triples(
                        relations=relations,
                        text=text,
                        mention=mention,
                        return_all=return_all,
                    )
                )
                for triple in candidate_triples:
                    sw = triple.subject[0]
                    ow = triple.object[0]
                    pred = triple.predicate
                    # avoid duplicate triples for the same mention
                    kt = (sw, pred, ow)
                    curr_mention_triples[kt] = mention.span
                    curr_mention_triple_scores[kt] = (
                        candidate_triple_scores.get(triple, 0.0)
                    )
            if top_k is not None:
                # keep only top-k triples by span length
                curr_mention_triples = sorted(
                    curr_mention_triples.items(),
                    key=lambda x: curr_mention_triple_scores.get(x[0], 0.0),
                    reverse=True,
                )
                curr_mention_triples = dict(curr_mention_triples[:top_k])
            for triple, span in curr_mention_triples.items():
                collection[0].append(triple)
                collection[1].append(span)
        return collection

    def augment_text_with_triples(self, batch: dict) -> dict:
        batch_items = []
        for item in dict2iter(batch):
            item_text: str = item["text"]
            triple_spo, mention_spans = self.extract_triples(text=item_text)
            original_mention_strs = []
            for mention_span in mention_spans:
                original_mention_strs.append(
                    item_text[mention_span.start : mention_span.end]
                )
            injections = []
            for _spo, _span in zip(triple_spo, mention_spans):
                injection_text = " {} {} ".format(_spo[1], _spo[2])
                injection_position = _span.end
                injections.append(
                    Injection(injection_text, injection_position)
                )
            augmented_text, mention_spans, triple_spans = inject(
                item_text, injections, mention_spans
            )
            triples = []
            for i in range(len(mention_spans)):
                mention_span = mention_spans[i]
                triple_span = triple_spans[i]
                triple_str = augmented_text[
                    triple_span.start : triple_span.end
                ]
                original_mention_str_augmented = augmented_text[
                    mention_span.start : mention_span.end
                ]
                assert (
                    original_mention_strs[i] == original_mention_str_augmented
                ), "Original mention string does not match after augmentation."
                triples.append(({
                    "span.type": "char",
                    # triple data
                    "triple.subject": triple_spo[i][0],
                    "triple.predicate": triple_spo[i][1],
                    "triple.object": triple_spo[i][2],
                    # triple span data
                    "mention.span.start": mention_span.start,
                    "mention.span.end": mention_span.end,
                    "mention.text": original_mention_strs[i],
                    # triple span data
                    "triple.span.start": triple_span.start,
                    "triple.span.end": triple_span.end,
                    "triple.text": triple_str,
                }))
            item["text"] = augmented_text
            item["original_text"] = item_text
            item["triples"] = triples
            batch_items.append(item)
        return batch_items

    def base_tokenize(self, batch: dict) -> dict:
        if not isinstance(batch, dict):
            batch = iter2dict(batch)
        data = self.tokenizer(batch["text"], return_offsets_mapping=True)
        batch.update(data)
        # update the augnmented triples to use token indexes (if any)
        triples_list = []
        for item in dict2iter(batch):
            item["tokens"] = [
                self.tokenizer.convert_ids_to_tokens(tid)
                for tid in item["input_ids"]
            ]
            triples = get_token_aligned_triples(item)
            triples_list.append(triples)
        batch["triples"] = triples_list
        return batch

    def extract_token_tree(self, item: dict):
        input_ids = item["input_ids"]
        tree = TokenTree()
        for input_id in input_ids:
            text = self.convert_ids_to_tokens(input_id)
            tree.append(text)
        # mark parent nodes for triples by setting parent for each token in tree
        tree.head = 0
        tree.tail = len(input_ids) - 1
        for triple in item["triples"]:
            if triple["span.type"] != "token":
                raise ValueError(
                    "Triples must be token-aligned to extract token tree."
                )
            mention_end_token = triple["mention.span.end"] - 1
            if mention_end_token is None:
                raise ValueError("Mention end token is None.")
            triple_start_token = triple["triple.span.start"]
            assert mention_end_token is not None, "mention_end_token is None"
            # triplet start should be the child of the mention end token
            tree.nodes[triple_start_token]["parent"] = mention_end_token
            # triplet end + 1 should be the child of the mention end token
            triple_end_token = triple["triple.span.end"]
            if triple_end_token < len(input_ids):
                tree.nodes[triple_end_token]["parent"] = mention_end_token
        tree.build()
        return tree

    def extract_features(self, batch: dict) -> dict:
        visibility_matrices = []
        soft_position_indexes = []
        tree_data = []
        # we have to set the visibility matrix according to the injected triples
        for i, item in enumerate(dict2iter(batch)):
            visibility_matrix = extract_visibility_matrix(item)
            visibility_matrices.append(visibility_matrix)
            # plot visibility_matrices in here.parent / "outputs" / "visibility_matrices"
            # filename = here / "outputs" / "visibility_matrix_{}.png".format(i)
            # fig = plot_dot_matrix(visibility_matrix)
            # fig.savefig(filename)
            tree = self.extract_token_tree(item)
            # add tree to data -- you can use TokenTree.loads to load this back
            tree_data.append(tree.dumps())
            soft_position_index = extract_soft_position_index(tree)
            soft_position_indexes.append(soft_position_index)
            # save plot of the tree for debugging to here.parent / "outputs" / "token_trees"
            # filename = here / "outputs" / "token_tree_{}".format(i)
            # tree.graphviz().render(filename, view=False, format="png")
        batch["visibility_matrix"] = visibility_matrices
        batch["position_ids"] = soft_position_indexes
        batch["token_type_ids"] = batch.get("token_type_ids", None)
        batch["token_tree"] = tree_data
        return batch

    def truncate_batch(self, batch: dict) -> dict:
        if not self.config.truncate:
            return batch
        batch_input_ids = batch["input_ids"]
        batch_position_ids = batch["position_ids"]
        batch_visibility_matrix = batch["visibility_matrix"]
        # Manual Truncation
        # We must truncate ALL fields consistently to MAX_LENGTH
        truncated_input_ids = []
        truncated_position_ids = []
        truncated_vis_matrix = []
        for i in range(len(batch_input_ids)):
            # Get current sequence length
            seq_len = len(batch_input_ids[i])
            # Determine cut-off point
            cut_off = min(seq_len, self.config.sequence_length)
            # Truncate 1D sequences
            t_ids = batch_input_ids[i][:cut_off]
            t_pos = batch_position_ids[i][:cut_off]
            # Truncate 2D Visibility Matrix
            # The matrix is N x N. We must slice both rows and columns.
            # Convert to numpy for easy slicing if it's a list
            t_vis = np.array(batch_visibility_matrix[i])
            t_vis = t_vis[:cut_off, :cut_off]
            truncated_input_ids.append(t_ids)
            truncated_position_ids.append(t_pos)
            truncated_vis_matrix.append(t_vis)
        # Return the corrected batch
        batch["input_ids"] = truncated_input_ids
        batch["position_ids"] = truncated_position_ids
        batch["visibility_matrix"] = truncated_vis_matrix
        return batch

    def __call__(self, batch: dict) -> dict:
        batch = self.augment_text_with_triples(batch)
        batch = self.base_tokenize(batch)
        batch = self.extract_features(batch)
        batch = self.truncate_batch(batch)
        return batch

    def get_features(self, raw_documents: list[str], y: list | None = None):
        triples = []
        desc = "Extracting Triples"
        pbar = tqdm.tqdm(
            enumerate(zip(raw_documents, y)),
            total=len(raw_documents),
            desc=desc,
        )
        for idx, (text, label) in pbar:
            for triple in self.extract_triples(text=text, return_all=True)[0]:
                subject, predicate, object_ = triple
                triples.append({
                    "text": text,
                    "label": label,
                    "subject": subject,
                    "predicate": predicate,
                    "object": object_,
                    "idx": idx,
                    "predicate_object": f"{predicate}:{object_}",
                })
        df = pd.DataFrame(triples)
        # Calculate diversity (number of unique subjects per predicate-object)
        diversity = (
            df.groupby("predicate_object")["subject"]
            .nunique()
            .rename("diversity")
        )
        # Prepare data for feature scoring
        temp_df = (
            df.groupby("idx")
            .agg({"predicate_object": lambda x: list(x), "label": "first"})
            .reset_index()
        )
        # .astype(str) ?
        raw_y = temp_df["label"].values
        encoder = LabelEncoder()
        y = encoder.fit_transform(raw_y)
        # Extract features from the predicate-object lists  (treating them like words)
        is_discrete = self.config.featurizer == "count"
        if self.config.featurizer == "tfidf":
            featurizer = TfidfVectorizer(
                tokenizer=passthrough,
                preprocessor=passthrough,
                lowercase=False,
                token_pattern=None,
            )
        elif is_discrete:
            featurizer = CountVectorizer(
                tokenizer=passthrough,
                preprocessor=passthrough,
                lowercase=False,
                token_pattern=None,
            )
        else:
            msg = f"invalid featurizer: {self.config.featurizer}"
            raise ValueError(msg)
        X = featurizer.fit_transform(temp_df["predicate_object"])
        # Feature names
        feature_names = featurizer.get_feature_names_out()
        return {
            "X": X,
            "y": y,
            "feature_names": feature_names,
            "diversity": diversity,
            "is_discrete": is_discrete,
            "classes": encoder.classes_,
        }

    def fit(self, raw_documents: list[str], y: list | None = None):
        """
        Fit the tokenizer using feature scoring methods.

        Parameters
        ----------
        raw_documents : list[str]
            List of raw text documents.
        y : list | None
            List of labels for each document.
        """
        features = self.get_features(raw_documents, y)
        X = features["X"]
        y = features["y"]
        feature_names = features["feature_names"]
        diversity = features["diversity"]
        is_discrete = features["is_discrete"]
        # # Multiply counts by diversity to get weighted counts
        # diversity_weights = np.array([
        #     diversity.get(fn, 1.0) for fn in feature_names
        # ])
        # # Apply diversity weights
        # X = X.multiply(diversity_weights)
        # Calculate Chi-Square
        chi2_scores, p_values = chi2(X, y)
        # Calculate Mutual Information (Information Gain)
        mi_scores = mutual_info_classif(
            X,
            y,
            random_state=42,
            n_neighbors=5,
            discrete_features=is_discrete,
        )
        # Train LinearSVC to get feature importance via coefficients
        svm_scores = linear_svm(X, y)
        # Extract RFECV scores
        # rfecv_scores = extract_RFECV_SVM_scores(X, y)
        # Store the scores in a DataFrame
        doc_counts = (X > 0).sum(axis=0).tolist()[0]
        class_counts = {}
        for class_label in np.unique(y):
            class_mask = y == class_label
            class_count = X[class_mask].sum(axis=0).tolist()[0]
            class_counts[f"classes[{class_label}].count"] = class_count
        triple_scores = pd.DataFrame(
            {
                "doc_counts": doc_counts,
                "svm_score": svm_scores,
                "chi2_score": chi2_scores,
                "mutual_info": mi_scores,
                # "rfecv_score": rfecv_scores,  # very slow to compute
                **class_counts,
            },
            index=feature_names,
        )
        self.triple_scores = triple_scores.join(diversity)
        self.threshold = None
        if len(self.triple_scores) < self.config.threshold_top_k:
            return
        self.threshold = (
            self.get_score(self.triple_scores)
            .sort_values(ascending=False)
            .iloc[self.config.threshold_top_k - 1]
        ).item()

    def get_score(
        self, score_row: dict | pd.Series | pd.DataFrame
    ) -> float | pd.Series:
        scorer = self.config.scorer
        if scorer == "default":
            if "svm_score" in score_row:
                return score_row["svm_score"]
            elif "chi2_score" in score_row:
                return score_row["chi2_score"]
        return score_row[scorer]


def extract_RFECV_SVM_scores(
    X: np.ndarray,
    y: list | np.ndarray,
    min_features_to_select: int = 500,
) -> np.ndarray:
    """
    Extract feature importance scores using RFECV with a linear SVM.
    1. Initialize a linear SVM estimator.
    2. Set up RFECV with the estimator, specifying step size and cross-validation folds.
    3. Fit RFECV to the data.
    4. Retrieve the support mask and ranking of features.
    5. Convert rankings to scores: features selected (support=True) get scores
         inversely proportional to their rank; unselected features get a score of 0.
    """
    estimator = SVC(kernel="linear")
    selector = RFECV(
        estimator,
        step=250,
        cv=5,
        min_features_to_select=min_features_to_select,
    )
    selector.fit(X, y)
    rfecv_scores = 1.0 / selector.ranking_
    return rfecv_scores


def linear_svm(X: np.ndarray, y: list | np.ndarray) -> np.ndarray:
    clf = LinearSVC(max_iter=10000, dual="auto")
    clf.fit(X, y)
    # Extract feature scores from SVM coefficients
    # For binary classification: coef_ is shape (1, n_features)
    # For multiclass: coef_ is shape (n_classes, n_features)
    if clf.coef_.shape[0] == 1:
        # Binary classification: use absolute coefficients
        svm_scores = np.abs(clf.coef_[0])
    else:
        # Multiclass: use max absolute coefficient across classes
        svm_scores = np.abs(clf.coef_).max(axis=0)
    return svm_scores
