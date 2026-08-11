"""OmniAnomaly (Su et al., KDD 2019) — 논문 기반 단순화 재구현.

GRU 인코더 + 스텝별 잠재 z의 VAE, 점수 = 마지막 스텝 재구성 NLL.
원 논문과의 차이 (중요):
(1) planar normalizing flow 미구현 (대각 가우시안 posterior),
(2) linear Gaussian state-space prior 대신 표준정규 prior,
(3) 점수는 window 마지막 스텝의 recon NLL (원 논문과 동일 취지의 단순화).
원 저장소는 라이선스가 명시돼 있어도 TF1 의존이라 vendored하지 않았다.
"""

from __future__ import annotations

import torch
from torch import nn

from tsad_forge.models._torch import TorchDetector
from tsad_forge.models.registry import register_model


class _OmniNet(nn.Module):
    def __init__(self, d_in: int, hidden: int, latent: int):
        super().__init__()
        self.gru = nn.GRU(d_in, hidden, batch_first=True)
        self.mu = nn.Linear(hidden, latent)
        self.logvar = nn.Linear(hidden, latent)
        self.dec = nn.GRU(latent, hidden, batch_first=True)
        self.out_mu = nn.Linear(hidden, d_in)
        self.out_logvar = nn.Linear(hidden, d_in)

    def forward(self, x):  # [B, w, D]
        h, _ = self.gru(x)
        mu, logvar = self.mu(h), self.logvar(h).clamp(-8, 8)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        hd, _ = self.dec(z)
        return self.out_mu(hd), self.out_logvar(hd).clamp(-8, 8), mu, logvar


@register_model("omni_anomaly")
class OmniAnomalyDetector(TorchDetector):
    generation = "gen3"

    def __init__(self, seed: int = 0, window: int = 64, hidden: int = 64, latent: int = 8, **kw):
        super().__init__(seed=seed, window=window, hidden=hidden, latent=latent, **kw)
        self.hidden = hidden
        self.latent = latent

    def _build(self, n_features: int) -> nn.Module:
        self.net_ = _OmniNet(n_features, self.hidden, self.latent)
        return self.net_

    def _loss(self, batch: torch.Tensor) -> torch.Tensor:
        out_mu, out_logvar, mu, logvar = self.net_(batch)
        nll = 0.5 * (out_logvar + (batch - out_mu) ** 2 / out_logvar.exp()).sum(dim=(1, 2))
        kl = -0.5 * (1 + logvar - mu**2 - logvar.exp()).sum(dim=(1, 2))
        return (nll + kl).mean()

    def _window_scores(self, batch: torch.Tensor) -> torch.Tensor:
        out_mu, out_logvar, _, _ = self.net_(batch)
        # 마지막 스텝의 recon NLL (causal 점수)
        last = 0.5 * (
            out_logvar[:, -1] + (batch[:, -1] - out_mu[:, -1]) ** 2 / out_logvar[:, -1].exp()
        )
        return last.mean(dim=1)
