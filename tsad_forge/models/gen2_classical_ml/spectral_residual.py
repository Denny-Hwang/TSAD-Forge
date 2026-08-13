"""Spectral Residual (Ren et al., KDD 2019) — the algorithm behind Microsoft's
Azure Anomaly Detector production service. Paper-based own implementation.

Saliency detection transferred from vision to time series:
1. log-amplitude spectrum L(f) of the series
2. spectral residual R(f) = L(f) - h_q * L(f)   (moving average of the spectrum)
3. saliency map S = |IFFT(exp(R + i*phase))|
4. score = normalized deviation of S from its local mean

Differences from the paper: the (optional) estimated points appended at the series
end and the SR-CNN learned threshold are omitted — scores go to our thresholding
module instead (CLAUDE.md §3). Channels are processed independently, max-aggregated.
"""

from __future__ import annotations

import numpy as np

from tsad_forge.models.base import BaseDetector
from tsad_forge.models.registry import register_model


@register_model("spectral_residual")
class SpectralResidualDetector(BaseDetector):
    generation = "gen2"

    def __init__(self, seed: int = 0, q: int = 3, z: int = 21, **params) -> None:
        """q: 스펙트럼 이동평균 창, z: saliency 국소 평균 창 (논문 기본값 근사)."""
        super().__init__(seed=seed, q=q, z=z, **params)
        self.q = q
        self.z = z

    def fit(self, X: np.ndarray) -> SpectralResidualDetector:
        self._as_2d(X)  # SR은 test 자체에서 saliency 계산 (train 불필요)
        self._fitted = True
        return self

    def _saliency(self, x: np.ndarray) -> np.ndarray:
        eps = 1e-8
        f = np.fft.fft(x)
        amp = np.abs(f) + eps
        log_amp = np.log(amp)
        kernel = np.ones(self.q) / self.q
        avg = np.convolve(
            np.pad(log_amp, (self.q // 2, self.q - 1 - self.q // 2), "edge"), kernel, "valid"
        )
        residual = log_amp - avg
        sal = np.abs(np.fft.ifft(np.exp(residual) * f / amp))
        # 국소 평균 대비 상대 편차 (논문 식 (9))
        kz = np.ones(self.z) / self.z
        local = np.convolve(
            np.pad(sal, (self.z // 2, self.z - 1 - self.z // 2), "edge"), kz, "valid"
        )
        return np.abs(sal - local) / (local + eps)

    def score(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        X = self._as_2d(X)
        return np.max(np.column_stack([self._saliency(X[:, d]) for d in range(X.shape[1])]), axis=1)
