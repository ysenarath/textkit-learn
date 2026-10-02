from __future__ import annotations

import json
from collections.abc import Generator
from pathlib import Path
from typing import Any

import igraph as ig
import numpy as np
import shap
from scipy.sparse import issparse, spmatrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import chi2, mutual_info_classif
from tqdm import auto as tqdm
from typing_extensions import Self


class FeatureScorer:
    preferences: Any

    def fit(
        self,
        X: np.ndarray | spmatrix,
        y: np.ndarray,
        feature_names: list[str],
    ):
        raise NotImplementedError

    def score(self, features: list[str]) -> dict[str, float]:
        raise NotImplementedError

    def dump(self, path: Path | str) -> None:
        raise NotImplementedError

    @classmethod
    def load(cls, path: Path | str) -> Self:
        raise NotImplementedError


class StoreMixin:
    preferences: Any
    threshold: Any

    def get_params(self) -> dict[str, Any]:
        return None

    def dump(self, path: Path | str) -> None:
        base_path = Path(path)
        preferences_path = base_path / "preferences.json"
        preferences_path.parent.mkdir(parents=True, exist_ok=True)
        with open(preferences_path, "w") as f:
            json.dump(self.preferences, f, indent=4)
        if hasattr(self, "threshold") and self.threshold is not None:
            threshold_path = base_path / "threshold.json"
            with open(threshold_path, "w") as f:
                json.dump(self.threshold, f, indent=4)
        params = self.get_params()
        if params:
            params_path = base_path / "params.json"
            with open(params_path, "w") as f:
                json.dump(params, f, indent=4)

    @classmethod
    def load(cls, path: Path | str) -> Self:
        base_path = Path(path)
        # load params if any
        params_path = base_path / "params.json"
        if params_path.exists():
            with open(params_path, "r") as f:
                params = json.load(f)
            obj = cls(**params)
        else:
            obj = cls()
        preferences_path = base_path / "preferences.json"
        with open(preferences_path, "r") as f:
            preferences = json.load(f)
        obj.preferences = preferences
        threshold_path = base_path / "threshold.json"
        if threshold_path.exists():
            with open(threshold_path, "r") as f:
                obj.threshold = json.load(f)
        return obj


class Chi2FeatureScorer(StoreMixin, FeatureScorer):
    def __init__(self):
        self.preferences: dict[str, float] = {}

    def fit(
        self, X: np.ndarray | spmatrix, y: np.ndarray, feature_names: list[str]
    ):
        chi2_values, _ = chi2(X, y)
        self.preferences = {
            feature_names[i]: chi2_values[i].item()
            for i in range(len(feature_names))
        }

    def score(self, features: list[str]) -> dict[str, float]:
        scores = {
            feature: self.preferences.get(feature, 0.0) for feature in features
        }
        # normalize scores
        total = sum(scores.values())
        if total > 0:
            scores = {k: v / total for k, v in scores.items()}
        return scores


class MutualInfoFeatureScorer(StoreMixin, FeatureScorer):
    def __init__(self, top_k: int | None = None):
        self.top_k = top_k
        self.preferences: dict[str, float] = {}
        self.threshold: float | None = None

    def get_params(self) -> dict[str, Any]:
        return {"top_k": self.top_k}

    def fit(
        self, X: np.ndarray | spmatrix, y: np.ndarray, feature_names: list[str]
    ):
        mi_values = mutual_info_classif(X, y, discrete_features="auto")
        self.preferences = {
            feature_names[i]: mi_values[i].item()
            for i in range(len(feature_names))
        }
        # set threshold to top 500 features if more than 500 features
        if self.top_k is not None and len(feature_names) > self.top_k:
            sorted_mi = sorted(self.preferences.values(), reverse=True)
            self.threshold = sorted_mi[self.top_k - 1]
        else:
            self.threshold = None

    def score(self, features: list[str]) -> dict[str, float]:
        scores = {}
        for feature in features:
            # feature: self.preferences.get(feature, 0.0)
            score = self.preferences.get(feature, 0.0)
            if self.threshold is not None and score < self.threshold:
                continue
            scores[feature] = score
        # normalize scores
        # total = sum(scores.values())
        # if total > 0:
        #     scores = {k: v / total for k, v in scores.items()}
        return scores


class ShapleyPageRankFeatureScorer(StoreMixin, FeatureScorer):
    def __init__(self):
        # preferences stores the minimum delta score between feature pairs
        #   featureA comes before featureB since A < B < C ...
        # preferences[featureA][featureB] = [float, ...]
        self.preferences: dict[str, dict[str, list[float]]] = {}

    def fit(
        self,
        X: np.ndarray | spmatrix,
        y: np.ndarray,
        feature_names: list[str],
    ):
        # train a simple random forest model
        model = RandomForestClassifier(
            n_estimators=100, max_depth=5, random_state=42
        )
        model.fit(X, y)
        explainer = shap.TreeExplainer(model)
        # batch processing to avoid memory issues of
        # - converting large sparse matrix to dense
        batch_size = 32
        num_samples = X.shape[0]
        for start in tqdm.trange(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            batch = X[start:end]
            self._update_preferences_batch(explainer, batch, feature_names)

    def _update_preferences_batch(
        self,
        explainer: shap.TreeExplainer,
        batch: np.ndarray | spmatrix,
        feature_names: list[str],
    ) -> None:
        if issparse(batch):
            batch = batch.toarray()
        # how to check if sparse matrix
        batch_shap_values: np.ndarray = explainer.shap_values(batch)
        # (batch_size, num_features, num_classes)
        feature_importance = np.abs(batch_shap_values).mean(axis=(0, 2))
        # select top features based on mean absolute shap values
        for shap_values in batch_shap_values:
            self._update_preferences_one(shap_values, feature_names)

    def _update_preferences_one(
        self,
        shap_values: np.ndarray,
        feature_names: list[str],
    ) -> None:
        abs_shap_values = np.abs(shap_values).max(axis=1)
        indexes = np.where(abs_shap_values > 1e-3)
        feature_scores = {}
        for (x,) in zip(*indexes):
            score = abs_shap_values[x].item()
            feature_name = feature_names[x]
            if feature_name not in feature_scores:
                feature_scores[feature_name] = {}
            feature_scores[feature_name] = score
        ordered_feature_names = sorted(feature_scores.keys())
        for i, fa in enumerate(ordered_feature_names):
            for j in range(i + 1, len(ordered_feature_names)):
                fb = ordered_feature_names[j]
                delta = feature_scores[fa] - feature_scores[fb]
                self.preferences.setdefault(fa, {})
                self.preferences[fa].setdefault(fb, [])
                self.preferences[fa][fb].append(delta)

    def _get_preferences(
        self, features: list[str]
    ) -> Generator[tuple[str, str, float], None, None]:
        ordered_features = sorted(features)
        for i, fa in enumerate(ordered_features):
            for j in range(i + 1, len(ordered_features)):
                fb = ordered_features[j]
                if fa not in self.preferences:
                    continue
                if fb not in self.preferences[fa]:
                    continue
                weight = np.mean(self.preferences[fa][fb]).item()
                winner, loser = fa, fb
                if weight < 0:
                    winner, loser = loser, winner
                    weight = -weight
                yield winner, loser, weight

    # def score(self, features: list[str]) -> dict[str, float]:
    #     # Initialize scores for all features to handle those with no preferences
    #     scores = {f: 0.0 for f in features}
    #     # Accumulate weights from the generator
    #     for winner, loser, weight in self._get_preferences(features):
    #         scores[winner] += weight
    #         # Optional: you could also subtract weight from the loser
    #         # scores[loser] -= weight
    #     # Shift scores to be positive (if using subtraction) and normalize
    #     min_score = min(scores.values())
    #     if min_score < 0:
    #         scores = {k: v - min_score for k, v in scores.items()}
    #     total = sum(scores.values())
    #     if total > 0:
    #         scores = {k: v / total for k, v in scores.items()}
    #     # sort scores in descending order
    #     scores = dict(
    #         sorted(scores.items(), key=lambda item: item[1], reverse=True)
    #     )
    #     return scores

    # def score(self, features: list[str]) -> dict[str, float]:
    #     # graph based scoring
    #     G = DiGraph()
    #     for winner, loser, weight in self._get_preferences(features):
    #         if G.has_edge(winner, loser):
    #             G[loser][winner]["weight"] += weight
    #         else:
    #             G.add_edge(loser, winner, weight=weight)
    #     # compute page rank as scores
    #     scores = nx.pagerank(G, weight="weight")
    #     # normalize scores
    #     total = sum(scores.values())
    #     if total > 0:
    #         scores = {k: v / total for k, v in scores.items()}
    #     # ensure all features are present in scores
    #     for feature in features:
    #         if feature not in scores:
    #             scores[feature] = 0.0
    #     return scores

    def score(self, features: list[str]) -> dict[str, float]:
        edges = []
        weights = []
        for winner, loser, weight in self._get_preferences(features):
            edges.append((loser, winner))
            weights.append(weight)
        g = ig.Graph(directed=True)
        g.add_vertices(features)
        g.add_edges(edges)
        g.es["weight"] = weights
        scores = g.pagerank(weights="weight")
        scores_dict = {g.vs[i]["name"]: scores[i] for i in range(len(g.vs))}
        # normalize scores
        total = sum(scores_dict.values())
        if total > 0:
            scores_dict = {k: v / total for k, v in scores_dict.items()}
        return scores_dict


def get_scorer(name: str, **kwargs) -> type[FeatureScorer]:
    if name == "rf_shapley_pagerank":
        return ShapleyPageRankFeatureScorer
    elif name == "chi2_score":
        return Chi2FeatureScorer
    elif name == "mutual_info" or name == "default":
        return MutualInfoFeatureScorer
    raise ValueError(name)
