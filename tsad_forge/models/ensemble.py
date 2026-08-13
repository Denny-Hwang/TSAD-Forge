"""ForgeEnsemble (proposed): rank-consensus of cheap, diverse baseline detectors.

Motivation (our proposal, honestly framed): across TSB-AD and our own lite results,
no single simple detector wins everywhere, but *different* simple detectors fail on
*different* entities. Outlier-ensemble theory (Aggarwal & Sathe, 2017) says
averaging normalized scores of diverse, weakly correlated detectors reduces variance
without new hyperparameters. This model averages the *rank-transformed* scores of
four cheap members spanning three mechanisms:

- sub_pca   (linear subspace reconstruction)
- sub_knn   (subsequence distance)
- iforest   (isolation / density)
- spectral_residual (frequency-domain saliency)

Rank transform makes members' score scales comparable without distribution
assumptions. Members run with their default configs; total cost stays close to the
slowest member. Classified under gen0 ("baselines") — it is a reference point, not
a novelty claim; any model on the leaderboard should at least beat it.
"""

from __future__ import annotations

import numpy as np

from tsad_forge.models.base import BaseDetector
from tsad_forge.models.registry import get_model, register_model

_MEMBERS = ("sub_pca", "sub_knn", "iforest", "spectral_residual")


@register_model("ensemble_simple")
class ForgeEnsemble(BaseDetector):
    generation = "gen0"

    def __init__(self, seed: int = 0, members: tuple[str, ...] = _MEMBERS, **params) -> None:
        super().__init__(seed=seed, members=list(members), **params)
        self.members = members

    def fit(self, X: np.ndarray) -> ForgeEnsemble:
        X = self._as_2d(X)
        self.models_ = [get_model(name, seed=self.seed) for name in self.members]
        for m in self.models_:
            m.fit(X)
        self._fitted = True
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        X = self._as_2d(X)
        T = len(X)
        ranks = np.zeros(T)
        for m in self.models_:
            s = m.score(X)
            order = s.argsort().argsort().astype(np.float64)  # rank transform
            ranks += order / max(T - 1, 1)
        return ranks / len(self.models_)
