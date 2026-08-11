"""Hotelling T² / PCA-T²·SPE / Sub-PCA — 자체 구현 (Hotelling 1947; Jackson & Mudholkar 1979)."""

from __future__ import annotations

import numpy as np

from tsad_forge.models._window import embed_windows, window_scores_to_points
from tsad_forge.models.base import BaseDetector
from tsad_forge.models.registry import register_model


@register_model("hotelling_t2")
class HotellingT2Detector(BaseDetector):
    """전차원 Hotelling T²: (x-μ)' Σ⁻¹ (x-μ). Σ는 릿지 정칙화한 train 공분산."""

    generation = "gen1"

    def __init__(self, seed: int = 0, ridge: float = 1e-6, **params) -> None:
        super().__init__(seed=seed, ridge=ridge, **params)
        self.ridge = ridge

    def fit(self, X: np.ndarray) -> HotellingT2Detector:
        X = self._as_2d(X)
        self.mean_ = X.mean(axis=0)
        cov = np.cov(X.T, ddof=1).reshape(X.shape[1], X.shape[1])
        cov += self.ridge * np.trace(cov) / max(X.shape[1], 1) * np.eye(X.shape[1])
        self.prec_ = np.linalg.pinv(cov)
        self._fitted = True
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        R = self._as_2d(X) - self.mean_
        return np.einsum("ti,ij,tj->t", R, self.prec_, R)


@register_model("pca_t2spe")
class PCAT2SPEDetector(BaseDetector):
    """PCA 부분공간의 T²(주성분 공간) + SPE/Q(잔차 공간) 결합 점수.

    T²와 SPE를 train 분포의 99% 분위수로 각각 정규화해 최대값을 취한다
    (단일 결합 통계량의 통상적 근사; Yue & Qin 2001 참조).
    """

    generation = "gen1"

    def __init__(self, seed: int = 0, var_ratio: float = 0.9, mode: str = "combined", **params):
        super().__init__(seed=seed, var_ratio=var_ratio, mode=mode, **params)
        self.var_ratio = var_ratio
        self.mode = mode

    def fit(self, X: np.ndarray) -> PCAT2SPEDetector:
        X = self._as_2d(X)
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        Z = (X - self.mean_) / self.std_
        U, S, Vt = np.linalg.svd(Z, full_matrices=False)
        var = S**2
        ratio = np.cumsum(var) / var.sum() if var.sum() > 0 else np.ones_like(var)
        k = int(np.searchsorted(ratio, self.var_ratio) + 1)
        k = min(max(k, 1), len(S) - 1) if len(S) > 1 else 1
        self.components_ = Vt[:k]  # [k, D]
        self.eigvals_ = np.maximum(var[:k] / max(len(Z) - 1, 1), 1e-12)
        t2, spe = self._stats(Z)
        self.t2_ref_ = max(np.quantile(t2, 0.99), 1e-12)
        self.spe_ref_ = max(np.quantile(spe, 0.99), 1e-12)
        self._fitted = True
        return self

    def _stats(self, Z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        scores = Z @ self.components_.T  # [T, k]
        t2 = ((scores**2) / self.eigvals_).sum(axis=1)
        recon = scores @ self.components_
        spe = ((Z - recon) ** 2).sum(axis=1)
        return t2, spe

    def score(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        Z = (self._as_2d(X) - self.mean_) / self.std_
        t2, spe = self._stats(Z)
        if self.mode == "t2":
            return t2
        if self.mode == "spe":
            return spe
        return np.maximum(t2 / self.t2_ref_, spe / self.spe_ref_)


@register_model("sub_pca")
class SubPCADetector(BaseDetector):
    """Sub-PCA: 슬라이딩 윈도우 부분수열의 PCA 재구성 오차 (TSB-AD 강력 baseline 계열)."""

    generation = "gen1"

    def __init__(self, seed: int = 0, window: int = 50, var_ratio: float = 0.9, **params):
        super().__init__(seed=seed, window=window, var_ratio=var_ratio, **params)
        self.window = window
        self.var_ratio = var_ratio

    def fit(self, X: np.ndarray) -> SubPCADetector:
        X = self._as_2d(X)
        W = embed_windows(X, min(self.window, len(X)))
        self.mean_ = W.mean(axis=0)
        Zc = W - self.mean_
        U, S, Vt = np.linalg.svd(Zc, full_matrices=False)
        var = S**2
        ratio = np.cumsum(var) / var.sum() if var.sum() > 0 else np.ones_like(var)
        k = int(np.searchsorted(ratio, self.var_ratio) + 1)
        self.components_ = Vt[: max(k, 1)]
        self._fitted = True
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        X = self._as_2d(X)
        w = min(self.window, len(X))
        W = embed_windows(X, w) - self.mean_
        recon = (W @ self.components_.T) @ self.components_
        err = ((W - recon) ** 2).mean(axis=1)
        return window_scores_to_points(err, len(X), w)
