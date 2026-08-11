"""DAGMM (Zong et al., ICLR 2018) — 논문 기반 재구현.

압축망(AE) 잠재 + 재구성 특징을 추정망(GMM)에 넣어 에너지를 점수로 사용.
원 논문과의 차이: (1) point 단위가 아닌 윈도우 평탄화 입력으로 일반화,
(2) 공분산 정칙화를 대각 성분 클램프로 단순화.
"""

from __future__ import annotations

import torch
from torch import nn

from tsad_forge.models._torch import TorchDetector
from tsad_forge.models.registry import register_model


class _DAGMM(nn.Module):
    def __init__(self, d_in: int, latent: int, n_gmm: int):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d_in, 60), nn.Tanh(), nn.Linear(60, latent))
        self.dec = nn.Sequential(nn.Linear(latent, 60), nn.Tanh(), nn.Linear(60, d_in))
        self.est = nn.Sequential(
            nn.Linear(latent + 2, 10),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Linear(10, n_gmm),
            nn.Softmax(dim=1),
        )
        self.n_gmm = n_gmm

    def features(self, x):
        z_c = self.enc(x)
        x_hat = self.dec(z_c)
        rel_euc = (x - x_hat).norm(dim=1) / (x.norm(dim=1) + 1e-8)
        cos = torch.cosine_similarity(x, x_hat, dim=1)
        z = torch.cat([z_c, rel_euc.unsqueeze(1), cos.unsqueeze(1)], dim=1)
        return x_hat, z

    def energy(self, z, gamma, eps: float = 1e-6):
        # GMM 파라미터 (배치 추정)
        n = gamma.sum(dim=0) + eps  # [K]
        phi = n / len(z)
        mu = (gamma.unsqueeze(2) * z.unsqueeze(1)).sum(dim=0) / n.unsqueeze(1)  # [K, d]
        diff = z.unsqueeze(1) - mu.unsqueeze(0)  # [B, K, d]
        var = (gamma.unsqueeze(2) * diff**2).sum(dim=0) / n.unsqueeze(1) + eps  # 대각 공분산
        log_prob = -0.5 * ((diff**2 / var.unsqueeze(0)).sum(dim=2) + var.log().sum(dim=1))
        log_prob = log_prob + phi.log().unsqueeze(0)
        return -torch.logsumexp(log_prob, dim=1), var


@register_model("dagmm")
class DAGMMDetector(TorchDetector):
    generation = "gen3"

    def __init__(
        self,
        seed: int = 0,
        window: int = 32,
        latent: int = 3,
        n_gmm: int = 4,
        lambda_energy: float = 0.1,
        lambda_cov: float = 0.005,
        **kw,
    ):
        super().__init__(
            seed=seed,
            window=window,
            latent=latent,
            n_gmm=n_gmm,
            lambda_energy=lambda_energy,
            lambda_cov=lambda_cov,
            **kw,
        )
        self.latent = latent
        self.n_gmm = n_gmm
        self.lambda_energy = lambda_energy
        self.lambda_cov = lambda_cov

    def _build(self, n_features: int) -> nn.Module:
        self.net_ = _DAGMM(self._effective_window * n_features, self.latent, self.n_gmm)
        return self.net_

    def _loss(self, batch: torch.Tensor) -> torch.Tensor:
        x = batch.flatten(1)
        x_hat, z = self.net_.features(x)
        gamma = self.net_.est(z)
        energy, var = self.net_.energy(z, gamma)
        recon = ((x - x_hat) ** 2).mean()
        return recon + self.lambda_energy * energy.mean() + self.lambda_cov * (1.0 / var).sum()

    def _window_scores(self, batch: torch.Tensor) -> torch.Tensor:
        x = batch.flatten(1)
        _, z = self.net_.features(x)
        gamma = self.net_.est(z)
        energy, _ = self.net_.energy(z, gamma)
        return energy
