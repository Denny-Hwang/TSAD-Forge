"""LSTM 계열 예측 기반 탐지기 — 논문 기반 재구현.

- lstm_ad (Malhotra et al., ESANN 2015): stacked LSTM 다중스텝 예측 오차.
  원 논문과의 차이: 오차의 다변량 가우시안 우도 대신 제곱오차 합을 점수로 사용
  (우도 파라미터 추정은 검증셋 필요 — 프로토콜 단순화).
- lstm_p (Telemanom형, Hundman et al., KDD 2018): 단일스텝 예측 + EWMA 오차 평활.
  원 논문과의 차이: nonparametric dynamic thresholding은 임계값 모듈(SPOT 등)이
  담당한다 (관심사 분리, CLAUDE.md §3).
"""

from __future__ import annotations

import torch
from torch import nn

from tsad_forge.models._torch import TorchDetector
from tsad_forge.models.registry import register_model


class _LSTMForecast(nn.Module):
    def __init__(self, d_in: int, hidden: int, horizon: int):
        super().__init__()
        self.lstm = nn.LSTM(d_in, hidden, num_layers=2, batch_first=True)
        self.head = nn.Linear(hidden, d_in * horizon)
        self.horizon = horizon
        self.d_in = d_in

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B, w, D]
        out, _ = self.lstm(x)
        pred = self.head(out[:, -1])  # 마지막 hidden으로 다음 horizon 예측
        return pred.view(-1, self.horizon, self.d_in)


class _LSTMBase(TorchDetector):
    generation = "gen3"
    horizon = 1

    def __init__(self, seed: int = 0, window: int = 64, hidden: int = 64, **kw):
        super().__init__(seed=seed, window=window, hidden=hidden, **kw)
        self.hidden = hidden

    def _build(self, n_features: int) -> nn.Module:
        self.net_ = _LSTMForecast(n_features, self.hidden, self.horizon)
        return self.net_

    def _split(self, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.horizon
        return batch[:, :-h], batch[:, -h:]

    def _loss(self, batch: torch.Tensor) -> torch.Tensor:
        ctx, target = self._split(batch)
        return ((self.net_(ctx) - target) ** 2).mean()

    def _window_scores(self, batch: torch.Tensor) -> torch.Tensor:
        ctx, target = self._split(batch)
        return ((self.net_(ctx) - target) ** 2).mean(dim=(1, 2))


@register_model("lstm_ad")
class LSTMADDetector(_LSTMBase):
    """Malhotra형: horizon>1 다중스텝 예측."""

    horizon = 4


@register_model("lstm_p")
class LSTMPredictorDetector(_LSTMBase):
    """Telemanom형: 단일스텝 예측 + EWMA 오차 평활."""

    horizon = 1

    def __init__(
        self, seed: int = 0, window: int = 64, hidden: int = 64, smooth: float = 0.1, **kw
    ):
        super().__init__(seed=seed, window=window, hidden=hidden, **kw)
        self.smooth = smooth
        self.params["smooth"] = smooth

    def score(self, X):
        import numpy as np

        raw = super().score(X)
        out = np.empty_like(raw)
        acc = raw[0]
        for i, v in enumerate(raw):  # EWMA 평활 (Telemanom §3.2)
            acc = self.smooth * v + (1 - self.smooth) * acc
            out[i] = acc
        return out
