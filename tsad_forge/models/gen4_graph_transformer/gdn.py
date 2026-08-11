"""GDN (Deng & Hooi, AAAI 2021) — 논문 기반 단순화 재구현.

센서 임베딩 코사인 유사도 top-k 그래프 + 그래프 어텐션 1층으로 다음 스텝 예측,
점수 = 센서별 예측 편차의 정규화 최대값.
원 논문과의 차이: (1) 그래프를 학습 중 고정하지 않고 매 forward마다 임베딩에서 재계산,
(2) 단변량(D=1) 입력은 그래프가 자명하므로 시간 윈도우 MLP 예측으로 동작.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from tsad_forge.models._torch import TorchDetector
from tsad_forge.models.registry import register_model


class _GDN(nn.Module):
    def __init__(self, n_sensors: int, window: int, embed_dim: int, topk: int):
        super().__init__()
        self.embed = nn.Parameter(torch.randn(n_sensors, embed_dim))
        self.w = nn.Linear(window, embed_dim, bias=False)  # 센서별 시계열 -> 특징
        self.att_a = nn.Parameter(torch.randn(2 * embed_dim))
        self.out = nn.Sequential(nn.Linear(embed_dim, 64), nn.ReLU(), nn.Linear(64, 1))
        self.topk = min(topk, n_sensors - 1) if n_sensors > 1 else 0
        self.n_sensors = n_sensors

    def forward(self, x):  # x: [B, w, D] -> 예측 [B, D]
        B, w, D = x.shape
        feat = self.w(x.transpose(1, 2))  # [B, D, e]
        if self.topk > 0:
            sim = F.normalize(self.embed, dim=1) @ F.normalize(self.embed, dim=1).T
            sim.fill_diagonal_(-torch.inf)
            adj = torch.zeros_like(sim)
            idx = sim.topk(self.topk, dim=1).indices
            adj.scatter_(1, idx, 1.0)
            # 어텐션 계수 (GATv1형): e_ij = LeakyReLU(a' [g_i || g_j])
            g = feat + self.embed.unsqueeze(0)  # [B, D, e]
            e = torch.einsum(
                "bie,e->bi",
                torch.cat(
                    [g.unsqueeze(2).expand(-1, -1, D, -1), g.unsqueeze(1).expand(-1, D, -1, -1)],
                    dim=-1,
                ).reshape(B, D * D, -1),
                self.att_a,
            ).view(B, D, D)
            e = F.leaky_relu(e).masked_fill(adj.unsqueeze(0) == 0, -torch.inf)
            alpha = F.softmax(e, dim=2)
            agg = torch.einsum("bij,bje->bie", alpha, feat)
        else:
            agg = feat
        return self.out(F.relu(agg * self.embed.unsqueeze(0))).squeeze(-1)  # [B, D]


@register_model("gdn")
class GDNDetector(TorchDetector):
    generation = "gen4"

    def __init__(self, seed: int = 0, window: int = 32, embed_dim: int = 32, topk: int = 5, **kw):
        super().__init__(seed=seed, window=window, embed_dim=embed_dim, topk=topk, **kw)
        self.embed_dim = embed_dim
        self.topk = topk

    def _build(self, n_features: int) -> nn.Module:
        self.net_ = _GDN(n_features, self._effective_window - 1, self.embed_dim, self.topk)
        return self.net_

    def _loss(self, batch: torch.Tensor) -> torch.Tensor:
        pred = self.net_(batch[:, :-1])
        return ((pred - batch[:, -1]) ** 2).mean()

    def _window_scores(self, batch: torch.Tensor) -> torch.Tensor:
        err = (self.net_(batch[:, :-1]) - batch[:, -1]).abs()  # [B, D]
        # 센서별 편차의 강건 정규화 후 최대 (논문 식 (9)-(10) 단순화)
        med = err.median(dim=0, keepdim=True).values
        iqr = err.quantile(0.75, dim=0, keepdim=True) - err.quantile(0.25, dim=0, keepdim=True)
        return ((err - med) / (iqr + 1e-6)).max(dim=1).values
