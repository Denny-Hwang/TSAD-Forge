"""실행 계측 (런타임/메모리) — 리뷰 P0-1/P0-2 반영.

설계:
- 런타임은 fit/score 각각을 perf_counter로만 감싼다. 계측 구간 안에서 프로파일러를
  켜지 않는다 (tracemalloc은 실행을 수 배 느리게 만들어 과거 runtime_s를 부풀렸다).
- 호스트 메모리는 백그라운드 스레드가 프로세스 RSS를 주기적으로 샘플링해
  "구간 시작 대비 피크 증가분"을 기록한다 (psutil, BSD-3). 샘플링 간격보다 짧은
  스파이크는 놓칠 수 있다 — 런타임 무오염과 맞바꾼 의도된 트레이드오프.
- 모델이 실제로 CUDA에서 실행된 경우에만 torch.cuda.max_memory_allocated로
  정확한 피크 VRAM을 함께 기록한다 (오버헤드 무시 가능).
"""

from __future__ import annotations

import threading

import psutil


class PeakRssSampler:
    """with 블록 동안 프로세스 RSS 피크 증가분(MB)을 샘플링한다.

    peak_mb = max(관측 RSS) - 진입 시점 RSS. 블록 종료 시점 RSS도 후보에 포함해
    샘플 스레드가 한 번도 돌지 못한 초단기 블록에서도 음수가 나오지 않게 한다.
    """

    def __init__(self, interval_s: float = 0.02) -> None:
        self.interval_s = interval_s
        self.peak_mb: float = 0.0

    def __enter__(self) -> PeakRssSampler:
        self._proc = psutil.Process()
        self._baseline = self._proc.memory_info().rss
        self._peak = self._baseline
        self._stop = threading.Event()

        def _sample() -> None:
            while not self._stop.wait(self.interval_s):
                try:
                    rss = self._proc.memory_info().rss
                except psutil.Error:  # pragma: no cover - 프로세스 종료 경합
                    return
                if rss > self._peak:
                    self._peak = rss

        self._thread = threading.Thread(target=_sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        self._peak = max(self._peak, self._proc.memory_info().rss)
        self.peak_mb = max(0.0, (self._peak - self._baseline) / 1e6)


def reset_cuda_peak() -> bool:
    """CUDA 피크 메모리 카운터를 리셋. CUDA 추적 가능 여부를 반환."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            return True
    except ImportError:
        pass
    return False


def cuda_peak_mb() -> float:
    """reset_cuda_peak 이후의 피크 VRAM(MB). CUDA 추적 중일 때만 호출한다."""
    import torch

    return torch.cuda.max_memory_allocated() / 1e6
