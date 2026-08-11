"""TimesNet(-AD) (Wu et al., ICLR 2023) — 논문 기반 단순화 재구현.

핵심 기제(FFT 주기 발견 → 1D→2D 변환 → 2D conv → 역변환) 보존, 재구성 기반 점수.
원 논문과의 차이: (1) TimesBlock 1개·단일 주기(top-1)만 사용,
(2) inception 대신 3x3 conv 2층, (3) 임베딩 차원 소형화 (8GB/CPU 기본).
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from tsad_forge.models._torch import TorchDetector
from tsad_forge.models.registry import register_model


class _TimesBlock(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(d_model, d_model, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(d_model, d_model, 3, padding=1),
        )

    def forward(self, x):  # [B, w, d]
        B, w, d = x.shape
        # FFT로 지배 주기 추정 (배치 평균 진폭)
        amp = torch.fft.rfft(x, dim=1).abs().mean(dim=(0, 2))
        amp[0] = 0
        k = int(amp.argmax().clamp(min=1))
        period = max(w // max(k, 1), 2)
        n_rows = (w + period - 1) // period
        pad = n_rows * period - w
        xp = F.pad(x, (0, 0, 0, pad)).view(B, n_rows, period, d).permute(0, 3, 1, 2)
        out = self.conv(xp).permute(0, 2, 3, 1).reshape(B, n_rows * period, d)[:, :w]
        return x + out


class _TimesNet(nn.Module):
    def __init__(self, d_in: int, d_model: int):
        super().__init__()
        self.embed = nn.Linear(d_in, d_model)
        self.block = _TimesBlock(d_model)
        self.head = nn.Linear(d_model, d_in)

    def forward(self, x):
        return self.head(self.block(self.embed(x)))


@register_model("timesnet")
class TimesNetDetector(TorchDetector):
    generation = "gen4"

    def __init__(self, seed: int = 0, window: int = 64, d_model: int = 32, **kw):
        super().__init__(seed=seed, window=window, d_model=d_model, **kw)
        self.d_model = d_model

    def _build(self, n_features: int) -> nn.Module:
        self.net_ = _TimesNet(n_features, self.d_model)
        return self.net_

    def _loss(self, batch: torch.Tensor) -> torch.Tensor:
        return ((self.net_(batch) - batch) ** 2).mean()

    def _window_scores(self, batch: torch.Tensor) -> torch.Tensor:
        return ((self.net_(batch) - batch) ** 2).mean(dim=(1, 2))
