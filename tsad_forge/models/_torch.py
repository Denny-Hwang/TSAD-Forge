"""Gen3+ 딥러닝 탐지기 공용 베이스.

설계 원칙 (CLAUDE.md §10-6):
- 기본 config는 8GB VRAM에서 동작 (여기서는 소형 hidden/epoch 기본값으로 보장)
- CUDA 있으면 사용, 없으면 CPU (동일 결과 보장은 시드 고정 범위 내)
- train 윈도우 수 상한(max_train_windows)으로 대형 데이터셋 학습 시간 제한
- peak VRAM은 runner가 torch.cuda.max_memory_allocated로 수집
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from tsad_forge.models._window import window_scores_to_points
from tsad_forge.models.base import BaseDetector


def select_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    max_train_windows = 10000

    def __init__(
        self,
        seed: int = 0,
        window: int = 64,
        epochs: int = 10,
        batch_size: int = 128,
        lr: float = 1e-3,
        stride: int = 1,
        **params,
    ) -> None:
        super().__init__(
            seed=seed,
            window=window,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            stride=stride,
            **params,
        )
        self.window = window
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.stride = stride
        self.device = select_device()

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

    def fit(self, X: np.ndarray):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        X = self._as_2d(X)
        self.n_features_ = X.shape[1]
        self._effective_window = min(self.window, len(X))
        wins = self._windows(X, stride=self.stride)
        if len(wins) > self.max_train_windows:
            idx = np.random.default_rng(self.seed).choice(
                len(wins), self.max_train_windows, replace=False
            )
            wins = wins[np.sort(idx)]
        self.model_ = self._build(self.n_features_).to(self.device)
        opt = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
        loader = DataLoader(
            TensorDataset(wins),
            batch_size=self.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.seed),
        )
        self.model_.train()
        for epoch in range(self.epochs):
            self._epoch_hook(epoch)
            for (batch,) in loader:
                opt.zero_grad()
                loss = self._loss(batch.to(self.device))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model_.parameters(), 5.0)
                opt.step()
        self._fitted = True
        return self

    @torch.no_grad()
    def score(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        X = self._as_2d(X)
        wins = self._windows(X)  # stride 1 — 전 지점 커버
        self.model_.eval()
        scores = []
        for i in range(0, len(wins), self.batch_size):
            batch = wins[i : i + self.batch_size].to(self.device)
            scores.append(self._window_scores(batch).cpu().numpy())
        ws = np.concatenate(scores)
        w = len(X) - len(ws) + 1
        return window_scores_to_points(ws.astype(np.float64), len(X), w)
