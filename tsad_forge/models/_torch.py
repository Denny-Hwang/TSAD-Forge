"""Gen3+ 딥러닝 탐지기 공용 베이스.

설계 원칙 (CLAUDE.md §10-6):
- 기본 config는 8GB VRAM에서 동작 (여기서는 소형 hidden/epoch 기본값으로 보장)
- CUDA 있으면 사용, 없으면 CPU (동일 결과 보장은 시드 고정 범위 내)
- train 윈도우 수 상한(max_train_windows)으로 대형 데이터셋 학습 시간 제한
- peak VRAM은 runner가 torch.cuda.max_memory_allocated로 수집

학습 옵션 (리뷰 P0 "DL 학습 부족"; 전부 생성자 인자 = config의 model_params):

| 인자 | 기본값(레거시) | 의미 |
|---|---|---|
| `val_ratio` | 0.0 | train 시계열의 **마지막** 비율을 시간순 검증 구간으로 분리 (윈도우 겹침 누수 없음) |
| `patience` | None | 검증 손실이 `patience` epoch 동안 `min_delta` 이상 개선되지 않으면 조기 종료, 최적 가중치 복원 (`val_ratio>0` 필요) |
| `lr_schedule` | None | `"cosine"`(epoch 단위 cosine annealing) / `"plateau"`(검증 손실 정체 시 LR×0.5) |
| `amp` | False | CUDA에서만: `"bf16"`(권장, Ampere+) / `"fp16"`(GradScaler) / True(=bf16). CPU에서는 무시 |
| `max_train_windows` | 10000 | 학습 윈도우 상한 (None = 무제한). 시드 고정 서브샘플 |
| `weight_decay` | 0.0 | Adam(W) weight decay (>0이면 AdamW) |
| `grad_clip` | 5.0 | gradient norm clip (None = 비활성) |
| `score_batch_size` | batch_size | 추론 배치 (학습보다 크게 잡아 GPU 활용) |
| `deterministic` | False | cuDNN deterministic + benchmark 비활성 (CUDA 재현성 ↑, 속도 ↓) |

기본값만 쓰면 이 옵션들이 생기기 전과 **같은 연산 순서·같은 난수 소비**로 학습한다
(tests/test_training_loop.py가 레거시 루프와 점수 일치를 검사). 옵션은 config의
model_params에 들어가므로 켜는 순간 config_hash가 달라진다 — 레거시 결과와 섞이지 않는다.

학습 진단은 `diagnostics_`(runner가 `diag_*` 지표로 저장)에 남는다:
`diag_epochs_run`, `diag_train_loss_last`, `diag_val_loss_best`, `diag_n_train_windows`.
"""

from __future__ import annotations

import contextlib
import copy
import math

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from tsad_forge.models._window import window_scores_to_points
from tsad_forge.models.base import BaseDetector

_LR_SCHEDULES = (None, "cosine", "plateau")


def select_device(device: str | None = None) -> torch.device:
    """None/"auto" -> CUDA 가능하면 cuda, 아니면 cpu. 그 외는 그대로 사용."""
    if device in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _amp_dtype(amp: bool | str | None) -> torch.dtype | None:
    if amp in (None, False, "none", "off"):
        return None
    if amp in (True, "bf16", "bfloat16"):
        return torch.bfloat16
    if amp in ("fp16", "float16"):
        return torch.float16
    raise ValueError(f"unknown amp '{amp}' (expected False | 'bf16' | 'fp16')")


class TorchDetector(BaseDetector):
    """윈도우 단위 학습/스코어링 공통 루프.

    서브클래스 구현:
    - _build(n_features): 모듈 생성 (self에 부착)
    - _loss(batch) -> Tensor: 학습 손실
    - _window_scores(batch) -> Tensor[B]: 윈도우별 이상 점수
    선택 구현:
    - _epoch_hook(epoch): epoch 시작 시 (USAD 등 phase 전환용)
    """

    generation = "gen3"
    uses_torch = True
    #: runner가 해석한 실행 장치를 생성자 인자 `device`로 넘겨도 되는 모델
    accepts_device = True
    max_train_windows: int | None = 10000

    def __init__(
        self,
        seed: int = 0,
        window: int = 64,
        epochs: int = 10,
        batch_size: int = 128,
        lr: float = 1e-3,
        stride: int = 1,
        device: str | None = None,
        val_ratio: float = 0.0,
        patience: int | None = None,
        min_delta: float = 0.0,
        lr_schedule: str | None = None,
        amp: bool | str = False,
        max_train_windows: int | None | str = "class",
        weight_decay: float = 0.0,
        grad_clip: float | None = 5.0,
        score_batch_size: int | None = None,
        deterministic: bool = False,
        **params,
    ) -> None:
        # 레거시 기본값과 다른 학습 옵션만 params(get_config)에 남긴다
        extra = {
            k: v
            for k, v, legacy in (
                ("val_ratio", val_ratio, 0.0),
                ("patience", patience, None),
                ("min_delta", min_delta, 0.0),
                ("lr_schedule", lr_schedule, None),
                ("amp", amp, False),
                ("max_train_windows", max_train_windows, "class"),
                ("weight_decay", weight_decay, 0.0),
                ("grad_clip", grad_clip, 5.0),
                ("score_batch_size", score_batch_size, None),
                ("deterministic", deterministic, False),
            )
            if v != legacy
        }
        # device는 결과 정체성에 runner가 따로 기록하므로 params(get_config)에는 넣지 않는다
        super().__init__(
            seed=seed,
            window=window,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            stride=stride,
            **extra,
            **params,
        )
        if not 0.0 <= val_ratio < 1.0:
            raise ValueError(f"val_ratio must be in [0, 1), got {val_ratio}")
        if patience is not None and val_ratio <= 0:
            raise ValueError("patience (early stopping) requires val_ratio > 0")
        if lr_schedule not in _LR_SCHEDULES:
            raise ValueError(f"unknown lr_schedule '{lr_schedule}' (expected {_LR_SCHEDULES})")
        if lr_schedule == "plateau" and val_ratio <= 0:
            raise ValueError("lr_schedule='plateau' requires val_ratio > 0")
        self.window = window
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.stride = stride
        self.device = select_device(device)
        self.val_ratio = float(val_ratio)
        self.patience = patience
        self.min_delta = float(min_delta)
        self.lr_schedule = lr_schedule
        self._amp = _amp_dtype(amp)
        if max_train_windows != "class":
            self.max_train_windows = max_train_windows  # type: ignore[assignment]
        self.weight_decay = float(weight_decay)
        self.grad_clip = grad_clip
        self.score_batch_size = int(score_batch_size or batch_size)
        self.deterministic = bool(deterministic)

    # --- 서브클래스 인터페이스 ---

    def _build(self, n_features: int) -> torch.nn.Module:
        raise NotImplementedError

    def _loss(self, batch: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _window_scores(self, batch: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _epoch_hook(self, epoch: int) -> None:
        pass

    # --- 공통 파이프라인 ---

    def _windows(self, X: np.ndarray, stride: int = 1) -> torch.Tensor:
        X = self._as_2d(X)
        w = min(self.window, len(X))
        view = np.lib.stride_tricks.sliding_window_view(X, w, axis=0)  # [N, D, w]
        arr = np.ascontiguousarray(view.transpose(0, 2, 1)[::stride])  # [N, w, D]
        return torch.from_numpy(arr).float()

    def _autocast(self):
        """CUDA + amp 설정일 때만 autocast. CPU에서는 아무것도 하지 않는다."""
        if self._amp is None or self.device.type != "cuda":
            return contextlib.nullcontext()
        return torch.autocast(device_type="cuda", dtype=self._amp)

    def _split_train_val(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        """시간순 분할: 앞 (1-val_ratio)는 학습, 뒤는 검증.

        윈도우를 만든 뒤 무작위로 나누면 겹치는 윈도우가 양쪽에 들어가 검증 손실이
        낙관적으로 편향된다. 시계열 자체를 먼저 자르면 누수가 없다.
        """
        if self.val_ratio <= 0:
            return X, None
        cut = int(round(len(X) * (1 - self.val_ratio)))
        w = min(self.window, len(X))
        if cut < w or len(X) - cut < w:
            # 너무 짧으면 검증을 포기하고 전체로 학습 (진단에 기록)
            return X, None
        return X[:cut], X[cut:]

    @torch.no_grad()
    def _eval_loss(self, wins: torch.Tensor) -> float:
        self.model_.eval()
        total, n = 0.0, 0
        for i in range(0, len(wins), self.score_batch_size):
            batch = wins[i : i + self.score_batch_size].to(self.device)
            with self._autocast():
                loss = self._loss(batch)
            total += float(loss) * len(batch)
            n += len(batch)
        self.model_.train()
        return total / max(n, 1)

    def fit(self, X: np.ndarray):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        if self.deterministic and self.device.type == "cuda":
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        X = self._as_2d(X)
        self.n_features_ = X.shape[1]
        self._effective_window = min(self.window, len(X))
        X_train, X_val = self._split_train_val(X)
        wins = self._windows(X_train, stride=self.stride)
        if self.max_train_windows is not None and len(wins) > self.max_train_windows:
            idx = np.random.default_rng(self.seed).choice(
                len(wins), self.max_train_windows, replace=False
            )
            wins = wins[np.sort(idx)]
        val_wins = self._windows(X_val) if X_val is not None else None
        self.model_ = self._build(self.n_features_).to(self.device)
        if self.weight_decay > 0:
            opt: torch.optim.Optimizer = torch.optim.AdamW(
                self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        else:
            opt = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
        sched: torch.optim.lr_scheduler.LRScheduler | None = None
        plateau: torch.optim.lr_scheduler.ReduceLROnPlateau | None = None
        if self.lr_schedule == "cosine":
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(self.epochs, 1))
        elif self.lr_schedule == "plateau":
            plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, factor=0.5, patience=max((self.patience or 4) // 2, 1)
            )
        # fp16만 loss scaling 필요 (bf16은 지수 범위가 fp32와 같다)
        use_scaler = self._amp is torch.float16 and self.device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda") if use_scaler else None
        loader = DataLoader(
            TensorDataset(wins),
            batch_size=self.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.seed),
        )

        best_val, best_state, bad_epochs = math.inf, None, 0
        epochs_run, last_train_loss = 0, math.nan
        self.model_.train()
        for epoch in range(self.epochs):
            self._epoch_hook(epoch)
            running, seen = 0.0, 0
            for (batch,) in loader:
                batch = batch.to(self.device)
                opt.zero_grad()
                with self._autocast():
                    loss = self._loss(batch)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                else:
                    loss.backward()
                if self.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model_.parameters(), self.grad_clip)
                if scaler is not None:
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()
                running += float(loss.detach()) * len(batch)
                seen += len(batch)
            epochs_run = epoch + 1
            last_train_loss = running / max(seen, 1)
            if sched is not None:
                sched.step()
            if val_wins is not None:
                val_loss = self._eval_loss(val_wins)
                if plateau is not None:
                    plateau.step(val_loss)
                if val_loss < best_val - self.min_delta:
                    best_val, bad_epochs = val_loss, 0
                    best_state = copy.deepcopy(self.model_.state_dict())
                else:
                    bad_epochs += 1
                if self.patience is not None and bad_epochs >= self.patience:
                    break
            if not math.isfinite(last_train_loss):
                raise FloatingPointError(
                    f"{type(self).__name__}: training loss became non-finite at epoch {epoch}"
                )
        if best_state is not None:
            self.model_.load_state_dict(best_state)  # 조기 종료 시 최적 epoch 가중치

        self.diagnostics_ = {
            "diag_epochs_run": float(epochs_run),
            "diag_train_loss_last": float(last_train_loss),
            "diag_n_train_windows": float(len(wins)),
        }
        if val_wins is not None:
            self.diagnostics_["diag_val_loss_best"] = float(best_val)
        self._fitted = True
        return self

    @torch.no_grad()
    def score(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        X = self._as_2d(X)
        wins = self._windows(X)  # stride 1 — 전 지점 커버
        self.model_.eval()
        scores = []
        for i in range(0, len(wins), self.score_batch_size):
            batch = wins[i : i + self.score_batch_size].to(self.device)
            with self._autocast():
                out = self._window_scores(batch)
            scores.append(out.float().cpu().numpy())
        ws = np.concatenate(scores)
        w = len(X) - len(ws) + 1
        return window_scores_to_points(ws.astype(np.float64), len(X), w)
