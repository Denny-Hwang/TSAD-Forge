"""MambaTSAD (Chen et al., IEEE SPL 2024, arXiv:2405.19823) — 논문 기반 재구현, 두 변형.

구조: HP filter(추세/순환 분해) → AMA(적응 이동평균) 평활 → selective SSM(Mamba형)
재구성 → 재구성 오차 점수.

**두 변형 (CLAUDE.md §3)** — 알려진 구현 이슈 4가지를 기준으로 나뉜다:

| 이슈 | faithful (원 구현 충실) | fixed (수정판) |
|---|---|---|
| (a) hidden state 인덱싱 | y_t = C_t·h_{t-1} (off-by-one) | y_t = C_t·h_t |
| (b) CPU/GPU 분기 | CPU 경로에서 선택성(Δ_t) 비활성(평균 Δ) | 장치 무관 동일 selective scan |
| (c) HP filter 목적 | 추세(trend)를 모델 입력으로 사용 (순환 성분을 버림) | 순환(cycle) 성분을 입력으로 사용 |
| (d) AMA 윈도우 | 전체 시계열 전역 FFT로 단일 주기 → 고정 창 | 윈도우별 국소 FFT로 적응 창 |

주의: 원 저장소 코드를 복사하지 않은 재구현이므로 'faithful'은 위 4개 이슈의
동작 특성을 재현한 근사이며, 수치가 원 구현과 동일함을 보장하지는 않는다.
비교 실험(benchmarks)에서 두 변형을 나란히 실행해 수정 효과를 정량화한다.

mamba-ssm CUDA 커널이 없어도 동작하는 순수 PyTorch selective scan을 기본 경로로
사용한다 (느리지만 동작 — CLAUDE.md §3).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from tsad_forge.models._torch import TorchDetector
from tsad_forge.models.registry import register_model


def hp_filter(x: np.ndarray, lam: float = 1600.0) -> tuple[np.ndarray, np.ndarray]:
    """Hodrick-Prescott filter: (trend, cycle) 반환. 채널 독립, 밴드 행렬 해법."""
    from scipy import sparse
    from scipy.sparse.linalg import spsolve

    T = len(x)
    if T < 4:
        return x.copy(), np.zeros_like(x)
    eye = sparse.eye(T, format="csc")
    D = sparse.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(T - 2, T), format="csc")
    trend = np.column_stack([spsolve(eye + lam * (D.T @ D), x[:, d]) for d in range(x.shape[1])])
    return trend, x - trend


def _dominant_period(x: np.ndarray, min_p: int = 2) -> int:
    """FFT 지배 주파수 → 주기 (없으면 min_p)."""
    x = x - x.mean()
    if len(x) < 2 * min_p or np.allclose(x, 0):
        return min_p
    amp = np.abs(np.fft.rfft(x))
    amp[0] = 0
    k = int(np.argmax(amp))
    return max(len(x) // k, min_p) if k > 0 else min_p


def ama_smooth(x: np.ndarray, global_fft: bool, chunk: int = 256) -> np.ndarray:
    """AMA(적응 이동평균): FFT 주기 기반 창 크기의 이동평균 평활.

    global_fft=True(faithful): 전체 시계열 FFT 1회 → 고정 창.
    False(fixed): chunk별 국소 FFT → 구간별 적응 창.
    """
    out = np.empty_like(x)
    T, D = x.shape
    for d in range(D):
        col = x[:, d]
        if global_fft:
            w = min(_dominant_period(col) // 2 + 1, max(T // 4, 1))
            out[:, d] = _moving_avg(col, max(w, 1))
        else:
            res = np.empty(T)
            for s in range(0, T, chunk):
                seg = col[max(0, s - chunk // 2) : s + chunk]  # 이력 포함 국소 추정
                w = min(_dominant_period(seg) // 2 + 1, max(len(seg) // 4, 1))
                sm = _moving_avg(col[max(0, s - w) : s + chunk], max(w, 1))
                res[s : s + chunk] = sm[-len(col[s : s + chunk]) :]
            out[:, d] = res
    return out


def _moving_avg(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return x.copy()
    pad = np.pad(x, (w // 2, w - 1 - w // 2), mode="edge")
    return np.convolve(pad, np.ones(w) / w, mode="valid")


class _SelectiveSSM(nn.Module):
    """순수 PyTorch selective scan (S6 단순화: 대각 A, 데이터 의존 Δ·B·C)."""

    def __init__(self, d_model: int, d_state: int):
        super().__init__()
        self.a_log = nn.Parameter(torch.log(torch.rand(d_model, d_state) * 0.9 + 0.1))
        self.delta_proj = nn.Linear(d_model, d_model)
        self.bc_proj = nn.Linear(d_model, 2 * d_state)
        self.d = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor, off_by_one: bool, selective: bool) -> torch.Tensor:
        # x: [B, w, d_model]
        B_, w, dm = x.shape
        A = -torch.exp(self.a_log)  # [dm, N] (음수 보장 — 안정성)
        delta = F.softplus(self.delta_proj(x))  # [B, w, dm]
        if not selective:  # faithful의 CPU 분기: 선택성 제거 (시퀀스 평균 Δ)
            delta = delta.mean(dim=1, keepdim=True).expand_as(delta)
        Bc, Cc = self.bc_proj(x).chunk(2, dim=-1)  # [B, w, N] each
        h = x.new_zeros(B_, dm, A.shape[1])
        ys = []
        for t in range(w):
            dA = torch.exp(delta[:, t].unsqueeze(-1) * A.unsqueeze(0))  # [B, dm, N]
            dBx = delta[:, t].unsqueeze(-1) * Bc[:, t].unsqueeze(1) * x[:, t].unsqueeze(-1)
            h_prev = h
            h = dA * h + dBx
            state = h_prev if off_by_one else h  # (a) hidden state 인덱싱 이슈
            ys.append((state * Cc[:, t].unsqueeze(1)).sum(dim=-1) + self.d * x[:, t])
        return torch.stack(ys, dim=1)


class _MambaBlock(nn.Module):
    def __init__(self, d_in: int, d_model: int, d_state: int):
        super().__init__()
        self.in_proj = nn.Linear(d_in, 2 * d_model)
        self.conv = nn.Conv1d(d_model, d_model, 4, padding=3, groups=d_model)
        self.ssm = _SelectiveSSM(d_model, d_state)
        self.out_proj = nn.Linear(d_model, d_in)

    def forward(self, x, off_by_one: bool, selective: bool):
        u, z = self.in_proj(x).chunk(2, dim=-1)
        u = self.conv(u.transpose(1, 2))[:, :, : x.shape[1]].transpose(1, 2)  # causal
        u = F.silu(u)
        y = self.ssm(u, off_by_one=off_by_one, selective=selective)
        return self.out_proj(y * F.silu(z))


class _MambaTSADBase(TorchDetector):
    generation = "gen5"
    faithful: bool = True

    def __init__(
        self,
        seed: int = 0,
        window: int = 64,
        d_model: int = 32,
        d_state: int = 8,
        hp_lambda: float = 1600.0,
        **kw,
    ):
        super().__init__(
            seed=seed, window=window, d_model=d_model, d_state=d_state, hp_lambda=hp_lambda, **kw
        )
        self.d_model = d_model
        self.d_state = d_state
        self.hp_lambda = hp_lambda

    def _build(self, n_features: int) -> nn.Module:
        self.net_ = _MambaBlock(n_features, self.d_model, self.d_state)
        return self.net_

    def _preprocess(self, X: np.ndarray) -> np.ndarray:
        X = self._as_2d(X)
        trend, cycle = hp_filter(X, lam=self.hp_lambda)
        # (c) HP filter 목적함수 이슈: faithful은 trend를, fixed는 cycle을 입력으로
        comp = trend if self.faithful else cycle
        # (d) AMA: faithful은 전역 FFT 고정 창, fixed는 국소 적응 창
        return ama_smooth(comp, global_fft=self.faithful)

    def fit(self, X: np.ndarray):
        return super().fit(self._preprocess(X))

    def score(self, X: np.ndarray) -> np.ndarray:
        return super().score(self._preprocess(X))

    def _forward(self, batch: torch.Tensor) -> torch.Tensor:
        # (b) CPU/GPU 분기 이슈: faithful은 CPU에서 selective 비활성
        selective = (not self.faithful) or batch.is_cuda
        return self.net_(batch, off_by_one=self.faithful, selective=selective)

    def _loss(self, batch: torch.Tensor) -> torch.Tensor:
        return ((self._forward(batch) - batch) ** 2).mean()

    def _window_scores(self, batch: torch.Tensor) -> torch.Tensor:
        return ((self._forward(batch) - batch) ** 2).mean(dim=(1, 2))


@register_model("mamba_tsad_faithful")
class MambaTSADFaithful(_MambaTSADBase):
    """원 구현의 알려진 이슈 4가지를 재현한 변형 (모듈 docstring 표 참조)."""

    faithful = True


@register_model("mamba_tsad_fixed")
class MambaTSADFixed(_MambaTSADBase):
    """이슈 수정판: 올바른 상태 인덱싱, 장치 무관 selective scan, cycle 입력, 국소 AMA."""

    faithful = False
