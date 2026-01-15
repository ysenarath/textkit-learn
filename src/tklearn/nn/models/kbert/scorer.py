from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import shap
from scipy.sparse import issparse, spmatrix
from sklearn.ensemble import RandomForestClassifier
from tqdm import auto as tqdm


class FeatureExtractor:
    def fit(
        self, X: np.ndarray | spmatrix, y: np.ndarray, feature_names: list[str]
    ):
        raise NotImplementedError

    def score(self, features: list[str]) -> pd.DataFrame:
        raise NotImplementedError

    def select(self, features: list[str], top_k: int) -> list[str]:
        raise NotImplementedError


class ShapleyFeatureExtractor(FeatureExtractor):
    def __init__(self):
        self.preferences: dict[str, dict[str, list[float]]] = {}

    def save(self, path: Path | str):
        raise NotImplementedError

    @staticmethod
    def load(path: str) -> ShapleyFeatureExtractor:
        raise NotImplementedError

    def fit(
        self,
        X: np.ndarray | spmatrix,
        y: np.ndarray,
        feature_names: list[str],
    ):
        # need to reset preferences_df keep the same preferences across multiple fits
        self._preferences_df = None
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
            self._update_preferences(explainer, batch, feature_names)

    def _update_preferences(
        self,
        explainer: shap.TreeExplainer,
        batch: np.ndarray | spmatrix,
        feature_names: list[str],
    ) -> None:
        if issparse(batch):
            batch = batch.toarray()
        # how to check if sparse matrix
        batch_shap_values: np.ndarray = explainer.shap_values(batch)
        # shap_values.shape -> (batch_size, num_features, num_classes)
        feature_importance = np.abs(batch_shap_values).mean(axis=(0, 2))
        # batch_feature_scores = []
        # (featureA, featureB) -> delta score
        for shap_values in batch_shap_values:
            feature_shap_values = np.abs(shap_values).max(axis=1)
            indexes = np.where(feature_shap_values > 0.0)
            feature_scores = {}
            for (x,) in zip(*indexes):
                score = feature_shap_values[x].item()
                feature_scores[feature_names[x]] = score
            # sort by feature name to have a consistent order
            feature_scores = sorted(feature_scores.items())
            k = len(feature_scores)
            for i in range(k):
                fa, score_a = feature_scores[i]
                for j in range(i + 1, k):
                    fb, score_b = feature_scores[j]
                    diff = score_a - score_b
                    self.preferences.setdefault(fa, {})
                    self.preferences[fa].setdefault(fb, [])
                    self.preferences[fa][fb].append(diff)

    def select_preferences(self, features: list[str]):
        features = sorted(features)
        for i in range(len(features)):
            fa = features[i]
            for j in range(i + 1, len(features)):
                fb = features[j]
                scores = self.preferences.get(fa, {}).get(fb, [])
                yield {
                    "fa": fa,
                    "fb": fb,
                    "ds": np.mean(scores) if scores else 0.0,
                }

    def _graph_score(
        self, features: list[str], centrality: str = "betweenness_centrality"
    ) -> dict[str, float]:
        dG = nx.DiGraph()
        for row in self.select_preferences(features):
            fa, fb, weight = row["fa"], row["fb"], row["ds"]
            if weight < 0:
                dG.add_edge(fb, fa, weight=-weight)
            else:
                dG.add_edge(fa, fb, weight=weight)
        scores = (
            pd.DataFrame.from_dict(
                getattr(nx.centrality, centrality)(dG),
                orient="index",
                columns=["score"],
            )
            .reset_index(names=["feature_name"])
            .sort_values("score", ascending=False)
            .set_index("feature_name")
        )
        return scores["score"].to_dict()

    def _wins_score(self, features: list[str]) -> dict[str, float]:
        # count wins
        wins: dict[str, float] = {}
        for row in self.select_preferences(features):
            winner = row["fa"] if row["ds"] > 0 else row["fb"]
            if winner not in wins:
                wins[winner] = 0.0
            wins[winner] += 1.0
        wins = {
            k: v
            for k, v in sorted(
                wins.items(), key=lambda item: item[1], reverse=True
            )
        }
        return wins

    def score(self, features: list[str]) -> dict[str, float]:
        return self.wins_score(features)
