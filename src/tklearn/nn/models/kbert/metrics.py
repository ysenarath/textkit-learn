from __future__ import annotations

import json
from collections.abc import Generator
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import shap
from networkx import DiGraph
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


class PreferencesStoreMixin:
    preferences: Any

    def dump(self, path: Path | str) -> None:
        path = Path(path) / "preferences.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.preferences, f, indent=4)

    @classmethod
    def load(cls, path: Path | str) -> Self:
        path = Path(path) / "preferences.json"
        with open(path, "r") as f:
            preferences = json.load(f)
        obj = cls()
        obj.preferences = preferences
        return obj


class Chi2FeatureScorer(PreferencesStoreMixin, FeatureScorer):
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


class MutualInfoFeatureScorer(PreferencesStoreMixin, FeatureScorer):
    def __init__(self):
        self.preferences: dict[str, float] = {}

    def fit(
        self, X: np.ndarray | spmatrix, y: np.ndarray, feature_names: list[str]
    ):
        mi_values = mutual_info_classif(X, y, discrete_features="auto")
        self.preferences = {
            feature_names[i]: mi_values[i].item()
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


class ShapleyPageRankFeatureScorer(PreferencesStoreMixin, FeatureScorer):
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

    def score(self, features: list[str]) -> dict[str, float]:
        # graph based scoring
        G = DiGraph()
        for winner, loser, weight in self._get_preferences(features):
            if G.has_edge(winner, loser):
                G[loser][winner]["weight"] += weight
            else:
                G.add_edge(loser, winner, weight=weight)
        # compute page rank as scores
        scores = nx.pagerank(G, weight="weight")
        # normalize scores
        total = sum(scores.values())
        if total > 0:
            scores = {k: v / total for k, v in scores.items()}
        # ensure all features are present in scores
        for feature in features:
            if feature not in scores:
                scores[feature] = 0.0
        return scores


def get_scorer(score: str) -> type[FeatureScorer]:
    if score == "rf_shapley_pagerank":
        return ShapleyPageRankFeatureScorer
    elif score == "chi2_score":
        return Chi2FeatureScorer
    elif score == "mutual_info" or score == "default":
        return MutualInfoFeatureScorer
    raise ValueError(score)
