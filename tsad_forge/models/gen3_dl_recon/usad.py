"""USAD (Audibert et al., KDD 2020) — 논문 기반 재구현.

공유 인코더 + 두 디코더의 2단계 적대적 학습:
loss1 = (1/n)|x-AE1(x)|² + (1-1/n)|x-AE2(AE1(x))|²
loss2 = (1/n)|x-AE2(x)|² - (1-1/n)|x-AE2(AE1(x))|²
점수 = α|x-AE1(x)|² + β|x-AE2(AE1(x))|² (α=β=0.5 기본).
원 논문과의 차이: 디코더별 optimizer 분리 대신 epoch 내 교대 손실로 단순화.
"""

from __future__ import annotations

import torch
from torch import nn

from tsad_forge.models._torch import TorchDetector
from tsad_forge.models.registry import register_model


class _USAD(nn.Module):
    def __init__(self, d_in: int, latent: int):
        super().__init__()
        h = max(d_in // 2, latent * 2)
        self.enc = nn.Sequential(nn.Linear(d_in, h), nn.ReLU(), nn.Linear(h, latent), nn.ReLU())
        self.dec1 = nn.Sequential(nn.Linear(latent, h), nn.ReLU(), nn.Linear(h, d_in))
        self.dec2 = nn.Sequential(nn.Linear(latent, h), nn.ReLU(), nn.Linear(h, d_in))


@register_model("usad")
class USADDetector(TorchDetector):
    generation = "gen3"

    def __init__(self, seed: int = 0, window: int = 32, latent: int = 16, alpha: float = 0.5, **kw):
        super().__init__(seed=seed, window=window, latent=latent, alpha=alpha, **kw)
        self.latent = latent
        self.alpha = alpha
        self._epoch = 0

    def _build(self, n_features: int) -> nn.Module:
        self.net_ = _USAD(self._effective_window * n_features, self.latent)
        return self.net_

    def _epoch_hook(self, epoch: int) -> None:
        self._epoch = epoch + 1  # 1-indexed (1/n 가중)

    def _loss(self, batch: torch.Tensor) -> torch.Tensor:
        x = batch.flatten(1)
        n = self._epoch
        w1 = self.net_.dec1(self.net_.enc(x))
        w2 = self.net_.dec2(self.net_.enc(x))
        w3 = self.net_.dec2(self.net_.enc(w1))
        loss1 = (1 / n) * ((x - w1) ** 2).mean() + (1 - 1 / n) * ((x - w3) ** 2).mean()
        loss2 = (1 / n) * ((x - w2) ** 2).mean() - (1 - 1 / n) * ((x - w3) ** 2).mean()
        return loss1 + loss2

    def _window_scores(self, batch: torch.Tensor) -> torch.Tensor:
        x = batch.flatten(1)
        w1 = self.net_.dec1(self.net_.enc(x))
        w3 = self.net_.dec2(self.net_.enc(w1))
        return self.alpha * ((x - w1) ** 2).mean(dim=1) + (1 - self.alpha) * ((x - w3) ** 2).mean(
            dim=1
        )
