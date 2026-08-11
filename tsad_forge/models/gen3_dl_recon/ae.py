"""Dense Autoencoder — 윈도우 평탄화 재구성 오차 (Sakurada & Yairi 2014 계열)."""

from __future__ import annotations

import torch
from torch import nn

from tsad_forge.models._torch import TorchDetector
from tsad_forge.models.registry import register_model


@register_model("ae")
class AEDetector(TorchDetector):
    generation = "gen3"

    def __init__(self, seed: int = 0, window: int = 64, hidden: int = 64, latent: int = 16, **kw):
        super().__init__(seed=seed, window=window, hidden=hidden, latent=latent, **kw)
        self.hidden = hidden
        self.latent = latent

    def _build(self, n_features: int) -> nn.Module:
        d_in = self._effective_window * n_features
        self.net_ = nn.Sequential(
            nn.Linear(d_in, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, self.latent),
            nn.ReLU(),
            nn.Linear(self.latent, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, d_in),
        )
        return self.net_

    def _loss(self, batch: torch.Tensor) -> torch.Tensor:
        flat = batch.flatten(1)
        return ((self.net_(flat) - flat) ** 2).mean()

    def _window_scores(self, batch: torch.Tensor) -> torch.Tensor:
        flat = batch.flatten(1)
        return ((self.net_(flat) - flat) ** 2).mean(dim=1)
