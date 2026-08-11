"""TranAD (Tuli et al., VLDB 2022; 원저장소 BSD-3 확인) — 논문 기반 재구현.

인코더-디코더 transformer의 2-phase 적대적 자기조건화:
phase1 재구성 O1, phase1 오차를 focus score로 재입력한 phase2 재구성 O2.
loss = ε^-n |O1-W|² + (1-ε^-n)|O2-W|² (self-conditioning 적대 학습 단순화판).
점수 = 0.5|O1-W|² + 0.5|O2-W|².
원 논문과의 차이: 두 디코더의 min-max 게임을 epoch 가중 결합 손실로 단순화.
"""

from __future__ import annotations

import torch
from torch import nn

from tsad_forge.models._torch import TorchDetector
from tsad_forge.models.registry import register_model


class _TranAD(nn.Module):
    def __init__(self, d_in: int, d_model: int, window: int):
        super().__init__()
        self.proj = nn.Linear(2 * d_in, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead=4, dim_feedforward=2 * d_model, batch_first=True, dropout=0.1
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=1)
        self.dec1 = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_in))
        self.dec2 = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_in))
        pos = torch.arange(window).float()
        self.register_buffer("pos_enc", torch.sin(pos / 10.0).unsqueeze(0).unsqueeze(2))

    def forward(self, w):  # [B, win, D]
        focus = torch.zeros_like(w)
        h1 = self.encoder(self.proj(torch.cat([w, focus], dim=2)) + self.pos_enc)
        o1 = self.dec1(h1)
        focus2 = (o1 - w) ** 2  # self-conditioning
        h2 = self.encoder(self.proj(torch.cat([w, focus2], dim=2)) + self.pos_enc)
        o2 = self.dec2(h2)
        return o1, o2


@register_model("tranad")
class TranADDetector(TorchDetector):
    generation = "gen4"

    def __init__(self, seed: int = 0, window: int = 32, d_model: int = 64, **kw):
        super().__init__(seed=seed, window=window, d_model=d_model, **kw)
        self.d_model = d_model
        self._epoch = 1

    def _build(self, n_features: int) -> nn.Module:
        self.net_ = _TranAD(n_features, self.d_model, self._effective_window)
        return self.net_

    def _epoch_hook(self, epoch: int) -> None:
        self._epoch = epoch + 1

    def _loss(self, batch: torch.Tensor) -> torch.Tensor:
        o1, o2 = self.net_(batch)
        eps_n = 0.95**self._epoch
        return eps_n * ((o1 - batch) ** 2).mean() + (1 - eps_n) * ((o2 - batch) ** 2).mean()

    def _window_scores(self, batch: torch.Tensor) -> torch.Tensor:
        o1, o2 = self.net_(batch)
        return 0.5 * ((o1 - batch) ** 2).mean(dim=(1, 2)) + 0.5 * ((o2 - batch) ** 2).mean(
            dim=(1, 2)
        )
