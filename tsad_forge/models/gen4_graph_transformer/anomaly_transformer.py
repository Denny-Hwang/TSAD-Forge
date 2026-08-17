"""Anomaly Transformer (Xu et al., ICLR 2022) — 논문 기반 단순화 재구현.

핵심 기제(association discrepancy)를 보존:
- series-association: self-attention 분포
- prior-association: 학습 가능한 σ의 거리 가우시안
- 점수 = softmax(-AssDis) ⊙ 재구성 오차 (논문 식 (6))
원 논문과의 차이: (1) minimax 2단계 학습 대신 단일 손실(recon + AssDis 정칙화),
(2) 층 수 1, 소형 hidden (8GB/CPU 기본 config).
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from tsad_forge.models._torch import TorchDetector
from tsad_forge.models.registry import register_model


class _AnomalyAttention(nn.Module):
    def __init__(self, d_model: int, window: int):
        super().__init__()
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.sigma = nn.Linear(d_model, 1)
        dist = torch.abs(torch.arange(window)[:, None] - torch.arange(window)[None, :]).float()
        self.register_buffer("dist", dist)

    def forward(self, x):  # [B, w, d]
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        series = torch.softmax(q @ k.transpose(1, 2) / (q.shape[-1] ** 0.5), dim=2)
        sigma = F.softplus(self.sigma(x)) + 1e-3  # [B, w, 1]
        # self.dist는 register_buffer라 mypy가 Tensor로 좁히지 못함
        prior = torch.exp(-self.dist.unsqueeze(0) ** 2 / (2 * sigma**2))  # type: ignore[operator]
        prior = prior / prior.sum(dim=2, keepdim=True)
        return series @ v, series, prior


def _kl(p, q, eps=1e-8):
    return (p * ((p + eps).log() - (q + eps).log())).sum(dim=2)


class _AT(nn.Module):
    def __init__(self, d_in: int, d_model: int, window: int):
        super().__init__()
        self.proj = nn.Linear(d_in, d_model)
        self.att = _AnomalyAttention(d_model, window)
        self.ff = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.out = nn.Linear(d_model, d_in)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        h = self.proj(x)
        a, series, prior = self.att(h)
        h = self.norm1(h + a)
        h = self.norm2(h + self.ff(h))
        return self.out(h), series, prior


@register_model("anomaly_transformer")
class AnomalyTransformerDetector(TorchDetector):
    generation = "gen4"

    def __init__(self, seed: int = 0, window: int = 64, d_model: int = 64, lam: float = 3.0, **kw):
        super().__init__(seed=seed, window=window, d_model=d_model, lam=lam, **kw)
        self.d_model = d_model
        self.lam = lam

    def _build(self, n_features: int) -> nn.Module:
        self.net_ = _AT(n_features, self.d_model, self._effective_window)
        return self.net_

    def _loss(self, batch: torch.Tensor) -> torch.Tensor:
        recon, series, prior = self.net_(batch)
        ass_dis = (_kl(series, prior) + _kl(prior, series)).mean()
        return ((recon - batch) ** 2).mean() - self.lam * ass_dis * 0.01

    def _window_scores(self, batch: torch.Tensor) -> torch.Tensor:
        recon, series, prior = self.net_(batch)
        ass_dis = _kl(series, prior) + _kl(prior, series)  # [B, w]
        weight = torch.softmax(-ass_dis, dim=1)  # 논문 식 (6)
        err = ((recon - batch) ** 2).mean(dim=2)  # [B, w]
        return (weight * err).sum(dim=1)
