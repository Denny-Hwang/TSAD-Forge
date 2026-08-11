"""Donut형 윈도우 VAE (Xu et al., WWW 2018) — 논문 기반 재구현.

점수: 재구성 확률의 음수 (MC 표본 평균 근사).
원 논문과의 차이: (1) modified ELBO(결측 주입)와 MCMC imputation 미구현,
(2) 단변량 전용이던 원 방법을 다변량 평탄화 윈도우로 일반화.
"""

from __future__ import annotations

import torch
from torch import nn

from tsad_forge.models._torch import TorchDetector
from tsad_forge.models.registry import register_model


class _VAE(nn.Module):
    def __init__(self, d_in: int, hidden: int, latent: int):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU())
        self.mu = nn.Linear(hidden, latent)
        self.logvar = nn.Linear(hidden, latent)
        self.dec = nn.Sequential(nn.Linear(latent, hidden), nn.ReLU())
        self.out_mu = nn.Linear(hidden, d_in)
        self.out_logvar = nn.Linear(hidden, d_in)

    def forward(self, x):
        h = self.enc(x)
        mu, logvar = self.mu(h), self.logvar(h).clamp(-8, 8)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        hd = self.dec(z)
        return self.out_mu(hd), self.out_logvar(hd).clamp(-8, 8), mu, logvar


@register_model("vae_donut")
class DonutVAEDetector(TorchDetector):
    generation = "gen3"

    def __init__(
        self,
        seed: int = 0,
        window: int = 64,
        hidden: int = 100,
        latent: int = 8,
        n_samples: int = 8,
        **kw,
    ):
        super().__init__(
            seed=seed, window=window, hidden=hidden, latent=latent, n_samples=n_samples, **kw
        )
        self.hidden = hidden
        self.latent = latent
        self.n_samples = n_samples

    def _build(self, n_features: int) -> nn.Module:
        self.net_ = _VAE(self._effective_window * n_features, self.hidden, self.latent)
        return self.net_

    def _loss(self, batch: torch.Tensor) -> torch.Tensor:
        x = batch.flatten(1)
        out_mu, out_logvar, mu, logvar = self.net_(x)
        recon_nll = 0.5 * (out_logvar + (x - out_mu) ** 2 / out_logvar.exp()).sum(dim=1)
        kl = -0.5 * (1 + logvar - mu**2 - logvar.exp()).sum(dim=1)
        return (recon_nll + kl).mean()

    def _window_scores(self, batch: torch.Tensor) -> torch.Tensor:
        x = batch.flatten(1)
        nll = torch.zeros(len(x), device=x.device)
        for _ in range(self.n_samples):
            out_mu, out_logvar, _, _ = self.net_(x)
            nll += 0.5 * (out_logvar + (x - out_mu) ** 2 / out_logvar.exp()).mean(dim=1)
        return nll / self.n_samples
