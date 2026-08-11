"""DCdetector (Yang et al., KDD 2023) — 논문 기반 단순화 재구현.

이중 어텐션 대조 구조: patch-wise(패치 간) vs in-patch(패치 내) 어텐션 표현의
KL 불일치가 점수 (재구성 없음 — 순수 대조).
원 논문과의 차이: (1) 다중 패치 크기 대신 단일 patch_size,
(2) 채널 독립 처리 대신 채널 평균 표현, (3) stop-gradient 단순화.
"""

from __future__ import annotations

import torch
from torch import nn

from tsad_forge.models._torch import TorchDetector
from tsad_forge.models.registry import register_model


def _attn_dist(x: torch.Tensor) -> torch.Tensor:
    """self-attention 분포 [B, n, n]."""
    return torch.softmax(x @ x.transpose(1, 2) / (x.shape[-1] ** 0.5), dim=2)


class _DC(nn.Module):
    def __init__(self, d_in: int, d_model: int, window: int, patch: int):
        super().__init__()
        assert window % patch == 0
        self.patch = patch
        self.n_patch = window // patch
        self.proj_pw = nn.Linear(d_in * patch, d_model)  # patch-wise: 패치를 토큰으로
        self.proj_ip = nn.Linear(d_in, d_model)  # in-patch: 스텝을 토큰으로

    def forward(self, w):  # [B, win, D]
        B, win, D = w.shape
        patches = w.view(B, self.n_patch, self.patch * D)
        pw = _attn_dist(self.proj_pw(patches))  # [B, n_patch, n_patch]
        ip_tokens = self.proj_ip(w.view(B * self.n_patch, self.patch, D))
        ip = _attn_dist(ip_tokens).view(B, self.n_patch, self.patch, self.patch)
        # in-patch 어텐션을 패치 수준으로 업샘플: 패치 내 평균 주의 강도의 외적 근사
        ip_patch = ip.mean(dim=(2, 3))  # [B, n_patch]
        ip_patch = ip_patch.unsqueeze(2) * ip_patch.unsqueeze(1)
        ip_patch = ip_patch / (ip_patch.sum(dim=2, keepdim=True) + 1e-8)
        return pw, ip_patch


def _kl(p, q, eps=1e-8):
    return (p * ((p + eps).log() - (q + eps).log())).sum(dim=2)


@register_model("dcdetector")
class DCdetectorDetector(TorchDetector):
    generation = "gen4"

    def __init__(self, seed: int = 0, window: int = 64, d_model: int = 64, patch: int = 8, **kw):
        super().__init__(seed=seed, window=window, d_model=d_model, patch=patch, **kw)
        self.d_model = d_model
        self.patch = patch

    def _build(self, n_features: int) -> nn.Module:
        w = self._effective_window
        patch = self.patch
        while w % patch != 0:  # 짧은 시계열 방어
            patch -= 1
        self._patch_eff = patch
        self.net_ = _DC(n_features, self.d_model, w, patch)
        return self.net_

    def _loss(self, batch: torch.Tensor) -> torch.Tensor:
        pw, ip = self.net_(batch)
        # 대조 학습: 두 뷰의 상호 KL 최소화 (정상 데이터에서 일치 유도)
        return (_kl(pw, ip) + _kl(ip, pw)).mean()

    def _window_scores(self, batch: torch.Tensor) -> torch.Tensor:
        pw, ip = self.net_(batch)
        return (_kl(pw, ip) + _kl(ip, pw)).mean(dim=1)
