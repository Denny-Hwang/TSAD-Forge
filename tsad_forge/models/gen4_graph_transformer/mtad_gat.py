"""MTAD-GAT (Zhao et al., ICDM 2020) — 논문 기반 단순화 재구현.

특징 축 GAT + 시간 축 GAT → GRU → 예측 + 재구성 결합 점수.
원 논문과의 차이: (1) 재구성 경로를 VAE 대신 결정적 디코더로,
(2) 점수 결합은 gamma 가중 합(기본 0.5), (3) 1D conv 전처리 생략.
"""

from __future__ import annotations

import torch
from torch import nn

from tsad_forge.models._torch import TorchDetector
from tsad_forge.models.registry import register_model


def _self_attention(x: torch.Tensor) -> torch.Tensor:
    """단순 scaled dot-product self-attention (GAT 근사)."""
    a = torch.softmax(x @ x.transpose(1, 2) / (x.shape[-1] ** 0.5), dim=2)
    return a @ x


class _MTADGAT(nn.Module):
    def __init__(self, d_in: int, window: int, hidden: int):
        super().__init__()
        self.gru = nn.GRU(3 * d_in, hidden, batch_first=True)
        self.forecast = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, d_in))
        self.recon = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, window * d_in)
        )
        self.window = window
        self.d_in = d_in

    def forward(self, x):  # [B, w, D]
        feat_att = _self_attention(x.transpose(1, 2)).transpose(1, 2)  # 특징 축
        time_att = _self_attention(x)  # 시간 축
        h, _ = self.gru(torch.cat([x, feat_att, time_att], dim=2))
        last = h[:, -1]
        return self.forecast(last), self.recon(last).view(-1, self.window, self.d_in)


@register_model("mtad_gat")
class MTADGATDetector(TorchDetector):
    generation = "gen4"

    def __init__(self, seed: int = 0, window: int = 32, hidden: int = 64, gamma: float = 0.5, **kw):
        super().__init__(seed=seed, window=window, hidden=hidden, gamma=gamma, **kw)
        self.hidden = hidden
        self.gamma = gamma

    def _build(self, n_features: int) -> nn.Module:
        self.net_ = _MTADGAT(n_features, self._effective_window - 1, self.hidden)
        return self.net_

    def _loss(self, batch: torch.Tensor) -> torch.Tensor:
        ctx, target = batch[:, :-1], batch[:, -1]
        pred, recon = self.net_(ctx)
        return ((pred - target) ** 2).mean() + ((recon - ctx) ** 2).mean()

    def _window_scores(self, batch: torch.Tensor) -> torch.Tensor:
        ctx, target = batch[:, :-1], batch[:, -1]
        pred, recon = self.net_(ctx)
        f_err = ((pred - target) ** 2).mean(dim=1)
        r_err = ((recon - ctx) ** 2).mean(dim=(1, 2))
        return self.gamma * f_err + (1 - self.gamma) * r_err
