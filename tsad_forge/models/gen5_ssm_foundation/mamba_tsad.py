"""MambaTSAD (Chen et al., IEEE SPL 2024, arXiv:2405.19823) — 공식 코드 동작 재구현 (v2).

기준: 공식 저장소 https://github.com/aaaceo890/mamba_tsad @ 08fec0531712 (2026-10-09).
그 저장소에는 **LICENSE 파일이 없다** → CLAUDE.md §10-2에 따라 코드를 복사하지 않고,
코드를 읽어 확인한 동작을 이 모듈에서 독자적으로 재작성했다 (THIRD_PARTY_NOTICES.md).
수치가 공식 구현과 비트 단위로 같다고 보장하지 않는다 — 구조·학습 프로토콜·아래 이슈의
동작을 일치시키는 것이 목표다.

구조 (공식 코드에서 확인):
1. HP pre-filter: 각 윈도우·채널에 HP형 평활 → trend. 모델 입력은 x - trend(순환),
   trend는 마지막에 출력에 다시 더한다.
2. TokenEmbedding: Conv1d(k=3, circular, bias 없음).
3. Mamba 블록 × 3: RMSNorm → S6 (expand 2, d_state 16, d_conv 4, dt_rank=ceil(d/16)).
   **residual 연결 없음**. 각 블록 뒤 AMA(FFT 주기 기반 이동평균) 분해: 잔차는 다음
   블록으로, trend는 누적.
4. 출력: Conv1d(누적 trend) + HP trend + Conv1d(seasonal). 손실/점수 = 재구성 MSE.
5. 학습: stride-1 윈도우(길이 100), **셔플 없이 순차 배치(128)**, 배치 간 hidden state
   (conv/ssm) 전달, AdamW(lr 1e-3, wd 1e-5), epoch≥4부터 lr×0.9 (5 epoch 단위 계단),
   7 epoch, 마지막 epoch 모델, grad clip 없음.
6. 점수: test를 **겹치지 않는 윈도우(배치 1)** 로 순차 처리하며 hidden state 전달,
   점별 MSE의 채널 평균. 마지막 부분 윈도우는 0으로 패딩 후 잘라낸다.

공식 코드에서 확인한 구현 이슈와 플래그 (faithful = 공식 동작, fixed = 수정):

| 이슈 | 공식 코드 동작 | 플래그 | fixed |
|---|---|---|---|
| (a) eval hidden state 인덱싱 | 추론 시 다음 윈도우로 넘기는 상태가 `hs[:, b-1]` — 시간축을 배치 크기로 인덱싱. 배치 1이면 **첫 시점** 상태가 전달된다 | `state_index="upstream"` | `"last"` (마지막 시점) |
| (b) CPU/GPU 분기 | 학습 루프의 CPU 분기도 `.cuda()`를 호출 → CPU에서 학습 불가 (공식 결과는 GPU 전용) | 없음 | 이 구현은 장치 무관 동일 연산 |
| (c) HP 목적함수 | `(I + 2λD₁ᵀD₁ + λD₂ᵀD₂) τ = y` (1차 차분 항이 추가된 비표준 목적), λ=1e4 | `hp_objective="upstream"` | `"standard"`: HP 정의 `(I + λD₂ᵀD₂) τ = y` |
| (d) AMA 주기 추정 | PSD를 **배치·채널 전체**에서 평균해 주기 1개 → 학습(배치 128)과 추론(배치 1)의 추정 모집단이 다르고, 같은 윈도우라도 배치 구성에 따라 결과가 바뀜 | `ama_period="batch"` | `"sample"`: 윈도우별 주기 |

각 플래그를 model_params로 개별 지정하면 이슈 하나씩의 효과를 분리 측정할 수 있다
(configs/mamba_ablation.yaml).

v1(이 저장소의 이전 구현)은 공식 코드 확인 없이 추정한 이슈(off-by-one 출력, CPU
선택성 비활성, trend 입력, 전역 FFT 입력 평활)를 구현했고 그중 다수가 공식 동작과
달랐다. v1 결과는 benchmarks/results/_superseded/mamba_tsad_v1/로 격리했다.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from tsad_forge.models.base import BaseDetector
from tsad_forge.models.registry import register_model

STATE_INDEX = ("upstream", "last")
HP_OBJECTIVES = ("upstream", "standard")
AMA_PERIODS = ("batch", "sample")


# --- HP filter ---------------------------------------------------------------


def hp_matrix(length: int, lam: float, objective: str = "standard") -> np.ndarray:
    """HP형 평활 시스템 행렬 A (trend = A⁻¹ y).

    standard: I + λD₂ᵀD₂ (Hodrick-Prescott 정의).
    upstream: I + 2λD₁ᵀD₁ + λD₂ᵀD₂ (공식 MambaTSAD 코드의 목적 — 이슈 (c)).
    """
    if objective not in HP_OBJECTIVES:
        raise ValueError(f"unknown hp objective '{objective}' (expected {HP_OBJECTIVES})")
    eye = np.eye(length)
    d2 = eye[:-2] - 2 * eye[1:-1] + eye[2:]
    A = eye + lam * d2.T @ d2
    if objective == "upstream":
        d1 = eye[:-1] - eye[1:]
        A = A + 2 * lam * d1.T @ d1
    return A


def hp_filter(
    x: np.ndarray, lam: float = 1600.0, objective: str = "standard"
) -> tuple[np.ndarray, np.ndarray]:
    """HP filter: (trend, cycle). 채널 독립, 희소 밴드 해법 (긴 시계열용)."""
    from scipy import sparse
    from scipy.sparse.linalg import spsolve

    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    T = len(x)
    if T < 4:
        return x.copy(), np.zeros_like(x)
    eye = sparse.eye(T, format="csc")
    d2 = sparse.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(T - 2, T), format="csc")
    A = eye + lam * (d2.T @ d2)
    if objective == "upstream":
        d1 = sparse.diags([1.0, -1.0], [0, 1], shape=(T - 1, T), format="csc")
        A = A + 2 * lam * (d1.T @ d1)
    elif objective != "standard":
        raise ValueError(f"unknown hp objective '{objective}'")
    trend = np.column_stack([spsolve(A.tocsc(), x[:, d]) for d in range(x.shape[1])])
    return trend, x - trend


# --- AMA (FFT 주기 기반 이동평균 분해) ---------------------------------------


def _moving_average(x: torch.Tensor, period: int) -> torch.Tensor:
    """x [B, L, C]의 시간축 이동평균 (커널 period, 가장자리 복제로 길이 유지)."""
    L = x.shape[1]
    p = max(1, min(int(period), L))
    avg = F.avg_pool1d(x.transpose(1, 2), kernel_size=p, stride=1).transpose(1, 2)
    front = (p - 1) // 2 + int(p % 2 == 0)
    back = (p - 1) // 2
    parts = [avg[:, :1].expand(-1, front, -1), avg, avg[:, -1:].expand(-1, back, -1)]
    return torch.cat(parts, dim=1)


def ama_decompose(x: torch.Tensor, mode: str) -> tuple[torch.Tensor, torch.Tensor]:
    """지배 주기의 이동평균으로 (잔차, trend) 분해.

    batch(공식): PSD를 배치·채널 평균 → 배치 전체에 주기 1개.
    sample(수정): PSD를 채널 평균 → 윈도우마다 자기 주기.
    주기 = int(L / f), f = argmax PSD over 주파수 bin 1..L//2-1 (DC 제외).
    """
    if mode not in AMA_PERIODS:
        raise ValueError(f"unknown ama_period '{mode}' (expected {AMA_PERIODS})")
    L = x.shape[1]
    psd = torch.fft.rfft(x.transpose(1, 2), dim=-1).abs() ** 2  # [B, C, F]
    band = psd[..., 1 : L // 2]
    if band.shape[-1] == 0:
        return x - x, x
    if mode == "batch":
        f = int(band.mean(dim=(0, 1)).argmax()) + 1
        trend = _moving_average(x, L // f)
    else:
        freqs = band.mean(dim=1).argmax(dim=-1) + 1  # [B]
        periods = (L // freqs).tolist()
        trend = torch.empty_like(x)
        for p in sorted(set(periods)):
            idx = [i for i, q in enumerate(periods) if q == p]
            trend[idx] = _moving_average(x[idx], p)
    return x - trend, trend


# --- Mamba (S6) ----------------------------------------------------------------


class _RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def linear_scan(a: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """h_t = a_t * h_{t-1} + x_t (h_{-1}=0)의 전 시점 해, 시간축 dim=1.

    Hillis-Steele 병렬 prefix scan: log2(L)번의 원소별 결합으로 L번의 순차 갱신을
    대체한다 (수학적으로 동일, 부동소수 반올림 순서만 다름). 윈도우 100이면 7단계.
    """
    a, x = a.clone(), x.clone()
    length, offset = a.shape[1], 1
    while offset < length:
        x = torch.cat([x[:, :offset], x[:, offset:] + a[:, offset:] * x[:, :-offset]], dim=1)
        a = torch.cat([a[:, :offset], a[:, offset:] * a[:, :-offset]], dim=1)
        offset *= 2
    return x


class _S6(nn.Module):
    """선택적 SSM 블록 (Gu & Dao 2023, 표준 구성). 순수 PyTorch 병렬 스캔 (linear_scan)."""

    def __init__(self, d_model: int, d_state: int, expand: int, d_conv: int):
        super().__init__()
        self.d_inner = d_model * expand
        self.d_conv = d_conv
        self.dt_rank = math.ceil(d_model / 16)
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv = nn.Conv1d(self.d_inner, self.d_inner, d_conv, groups=self.d_inner)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner)
        a = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(a))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(
        self, x: torch.Tensor, state: tuple | None, state_index: str
    ) -> tuple[torch.Tensor, tuple]:
        b, length, _ = x.shape
        u, gate = self.in_proj(x).chunk(2, dim=-1)
        u = u.transpose(1, 2)  # [b, di, l]
        conv_state, ssm_state = (None, None) if state is None else (state[0][:b], state[1][:b])
        k = self.d_conv
        hist = u.new_zeros(b, self.d_inner, k - 1) if conv_state is None else conv_state[..., 1:]
        v = F.silu(self.conv(torch.cat([hist, u], dim=-1)))  # causal, 길이 l
        # 다음 호출로 넘길 conv 이력 (공식 동작): 학습 중에는 각 윈도우의 첫 입력을
        # 이력 버퍼에 밀어 넣고, 추론 중에는 마지막 d_conv개 입력을 넘긴다.
        if self.training:
            prev = u.new_zeros(b, self.d_inner, k) if conv_state is None else conv_state
            new_conv = torch.cat([prev[..., 1:], u[..., :1]], dim=-1)
        else:
            new_conv = u[..., -k:]
        v = v.transpose(1, 2)  # [b, l, di]

        A = -torch.exp(self.A_log)  # [di, n]
        n = A.shape[1]
        dt, Bm, Cm = self.x_proj(v).split([self.dt_rank, n, n], dim=-1)
        delta = F.softplus(self.dt_proj(dt))  # [b, l, di]
        dA = torch.exp(delta.unsqueeze(-1) * A)  # [b, l, di, n]
        dBu = delta.unsqueeze(-1) * Bm.unsqueeze(2) * v.unsqueeze(-1)
        if ssm_state is not None:  # 초기 상태 h₋₁을 첫 입력 항에 흡수
            dBu = torch.cat([dBu[:, :1] + dA[:, :1] * ssm_state.unsqueeze(1), dBu[:, 1:]], dim=1)
        H = linear_scan(dA, dBu)  # [b, l, di, n]
        y = (H * Cm.unsqueeze(2)).sum(-1) + v * self.D
        if self.training or state_index == "last":
            new_ssm = H[:, -1]
        else:  # 이슈 (a): 공식 추론 경로는 시간축을 배치 크기로 인덱싱한다
            new_ssm = H[:, min(b - 1, length - 1)]
        out = self.out_proj(y * F.silu(gate))
        return out, (new_conv.detach(), new_ssm.detach())


class _MambaTSADNet(nn.Module):
    def __init__(self, d_in, window, d_model, d_state, expand, d_conv, n_blocks, flags):
        super().__init__()
        self.flags = flags
        if flags["pre_filter"]:
            A = hp_matrix(window, flags["hp_lambda"], flags["hp_objective"])
            self.register_buffer("hp_inv", torch.from_numpy(np.linalg.inv(A)).float())
        self.embed = nn.Conv1d(d_in, d_model, 3, padding=1, padding_mode="circular", bias=False)
        nn.init.kaiming_normal_(self.embed.weight, mode="fan_in", nonlinearity="leaky_relu")
        self.norms = nn.ModuleList(_RMSNorm(d_model) for _ in range(n_blocks))
        self.blocks = nn.ModuleList(_S6(d_model, d_state, expand, d_conv) for _ in range(n_blocks))
        self.seasonal_proj = nn.Conv1d(
            d_model, d_in, 3, padding=1, padding_mode="replicate", bias=False
        )
        self.trend_proj = nn.Conv1d(
            d_model, d_in, 3, padding=1, padding_mode="replicate", bias=False
        )

    def forward(self, x: torch.Tensor, states: list | None) -> tuple[torch.Tensor, list]:
        fl = self.flags
        if fl["pre_filter"]:
            hp_trend = torch.einsum("ij,bjd->bid", self.hp_inv, x)
            h = x - hp_trend
        else:
            hp_trend = torch.zeros_like(x)
            h = x
        h = self.embed(h.transpose(1, 2)).transpose(1, 2)
        states = states or [None] * len(self.blocks)
        new_states, trends = [], torch.zeros_like(h)
        for norm, block, st in zip(self.norms, self.blocks, states, strict=True):
            h, ns = block(norm(h), st, fl["state_index"])  # residual 없음 (공식 구조)
            if fl["decomp"]:
                h, tr = ama_decompose(h, fl["ama_period"])
                trends = trends + tr
            new_states.append(ns)
        seasonal = self.seasonal_proj(h.transpose(1, 2)).transpose(1, 2)
        out = seasonal + hp_trend
        if fl["decomp"]:
            out = out + self.trend_proj(trends.transpose(1, 2)).transpose(1, 2)
        return out, new_states


class _MambaTSADBase(BaseDetector):
    generation = "gen5"
    uses_torch = True
    accepts_device = True
    #: v2 = 공식 코드 확인 후 재작성. 결과 정체성(config_hash)에 반영된다.
    model_version = 2
    variant_flags: dict = {}

    def __init__(
        self,
        seed: int = 0,
        window: int = 100,
        d_model: int = 32,
        d_state: int = 16,
        expand: int = 2,
        d_conv: int = 4,
        n_blocks: int = 3,
        epochs: int = 7,
        batch_size: int = 128,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        lr_decay: float = 0.9,
        lr_step: int = 5,
        hp_lambda: float = 1e4,
        pre_filter: bool = True,
        decomp: bool = True,
        state_index: str | None = None,
        hp_objective: str | None = None,
        ama_period: str | None = None,
        device: str | None = None,
        **params,
    ):
        if params:
            # 공식 학습 프로토콜을 따르므로 TorchDetector 옵션(val_ratio, amp 등)은 받지
            # 않는다 — 조용히 무시되면 config에 적힌 설정과 실제 실행이 어긋난다.
            raise TypeError(f"{type(self).__name__} got unsupported options: {sorted(params)}")
        flags = {
            "state_index": state_index or self.variant_flags["state_index"],
            "hp_objective": hp_objective or self.variant_flags["hp_objective"],
            "ama_period": ama_period or self.variant_flags["ama_period"],
        }
        for key, allowed in (
            ("state_index", STATE_INDEX),
            ("hp_objective", HP_OBJECTIVES),
            ("ama_period", AMA_PERIODS),
        ):
            if flags[key] not in allowed:
                raise ValueError(f"{key} must be one of {allowed}, got {flags[key]!r}")
        super().__init__(
            seed=seed,
            window=window,
            d_model=d_model,
            d_state=d_state,
            expand=expand,
            d_conv=d_conv,
            n_blocks=n_blocks,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            lr_decay=lr_decay,
            lr_step=lr_step,
            hp_lambda=hp_lambda,
            pre_filter=pre_filter,
            decomp=decomp,
            **flags,
        )
        self.window = window
        self.arch = (d_model, d_state, expand, d_conv, n_blocks)
        self.epochs, self.batch_size, self.lr = epochs, batch_size, lr
        self.weight_decay, self.lr_decay, self.lr_step = weight_decay, lr_decay, lr_step
        self.flags = {**flags, "pre_filter": pre_filter, "decomp": decomp, "hp_lambda": hp_lambda}
        if device in (None, "auto"):
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

    def _build(self, n_features: int, window: int) -> None:
        self.net_ = _MambaTSADNet(n_features, window, *self.arch, self.flags).to(self.device)

    def fit(self, X: np.ndarray) -> _MambaTSADBase:
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        X = self._as_2d(X).astype(np.float32)
        w = min(self.window, len(X))
        if w < 5:
            raise ValueError(f"series too short for MambaTSAD (len={len(X)})")
        self.window_ = w
        self._build(X.shape[1], w)
        view = np.lib.stride_tricks.sliding_window_view(X, w, axis=0).transpose(0, 2, 1)
        wins = torch.from_numpy(np.ascontiguousarray(view))  # [N, w, D], 순서 유지
        opt = torch.optim.AdamW(self.net_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.net_.train()
        loss_val = math.nan
        for epoch in range(1, self.epochs + 1):  # 공식 코드는 1-indexed epoch
            for g in opt.param_groups:
                g["lr"] = self.lr * self.lr_decay ** ((epoch + 1) // self.lr_step)
            states = None  # 매 epoch 처음에는 상태 없음, 이후 배치 간 전달
            total, seen = 0.0, 0
            for i in range(0, len(wins), self.batch_size):
                batch = wins[i : i + self.batch_size].to(self.device)
                out, states = self.net_(batch, states)
                loss = F.mse_loss(out, batch)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += float(loss.detach()) * len(batch)
                seen += len(batch)
            loss_val = total / max(seen, 1)
            if not math.isfinite(loss_val):
                raise FloatingPointError(f"{type(self).__name__}: non-finite loss at epoch {epoch}")
        self.diagnostics_ = {
            "diag_epochs_run": float(self.epochs),
            "diag_train_loss_last": float(loss_val),
            "diag_n_train_windows": float(len(wins)),
        }
        self._fitted = True
        return self

    @torch.no_grad()
    def score(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        X = self._as_2d(X).astype(np.float32)
        T, w = len(X), self.window_
        n_win = -(-T // w)
        padded = np.zeros((n_win * w, X.shape[1]), dtype=np.float32)
        padded[:T] = X
        self.net_.eval()
        states = None
        errs = []
        for k in range(n_win):  # 겹치지 않는 윈도우, 배치 1, 상태 전달 (공식 추론 경로)
            win = torch.from_numpy(padded[k * w : (k + 1) * w][None]).to(self.device)
            out, states = self.net_(win, states)
            errs.append(((out - win) ** 2).mean(dim=-1)[0].cpu().numpy())
        return np.concatenate(errs)[:T].astype(np.float64)


@register_model("mamba_tsad_faithful")
class MambaTSADFaithful(_MambaTSADBase):
    """공식 코드 동작: 이슈 (a)(c)(d)를 그대로 재현 (모듈 docstring 표)."""

    variant_flags = {"state_index": "upstream", "hp_objective": "upstream", "ama_period": "batch"}


@register_model("mamba_tsad_fixed")
class MambaTSADFixed(_MambaTSADBase):
    """수정판: 마지막 시점 상태 전달, 표준 HP 목적, 윈도우별 AMA 주기."""

    variant_flags = {"state_index": "last", "hp_objective": "standard", "ama_period": "sample"}
