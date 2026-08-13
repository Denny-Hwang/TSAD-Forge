"""HBOS (Histogram-Based Outlier Score; Goldstein & Dengel 2012) — own implementation.

A widely used practical baseline (PyOD's fastest detector): per-feature histograms
estimated on train; score = sum over features of -log(density). Feature independence
is assumed — that is exactly its weakness on correlated channels, and its strength
in speed. Windowed embedding (window>1) supplies temporal context.
"""

from __future__ import annotations

import numpy as np

from tsad_forge.models._window import embed_windows, window_scores_to_points
from tsad_forge.models.base import BaseDetector
from tsad_forge.models.registry import register_model


@register_model("hbos")
class HBOSDetector(BaseDetector):
    generation = "gen2"

    def __init__(self, seed: int = 0, window: int = 10, n_bins: int = 20, **params) -> None:
        super().__init__(seed=seed, window=window, n_bins=n_bins, **params)
        self.window = window
        self.n_bins = n_bins

    def fit(self, X: np.ndarray) -> HBOSDetector:
        W = embed_windows(self._as_2d(X), min(self.window, len(X)))
        self.edges_ = []
        self.dens_ = []
        for j in range(W.shape[1]):
            hist, edges = np.histogram(W[:, j], bins=self.n_bins)
            dens = hist / max(hist.sum(), 1)
            self.edges_.append(edges)
            self.dens_.append(np.maximum(dens, 1e-6))  # 미관측 구간 바닥값
        self._fitted = True
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        X = self._as_2d(X)
        w = min(self.window, len(X))
        W = embed_windows(X, w)
        total = np.zeros(len(W))
        for j in range(W.shape[1]):
            idx = np.clip(np.searchsorted(self.edges_[j], W[:, j]) - 1, 0, self.n_bins - 1)
            total += -np.log(self.dens_[j][idx])
        return window_scores_to_points(total / W.shape[1], len(X), w)
