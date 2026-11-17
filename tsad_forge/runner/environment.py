"""실행 환경 식별 — 장치 해석 + 환경 지문 (재현성, CLAUDE.md §9).

왜 필요한가:
- 같은 config라도 CPU/GPU에서 수치가 달라질 수 있다 (cuDNN 비결정성, AMP, 그리고
  MambaTSAD faithful처럼 장치에 따라 동작이 분기하는 모델). 장치가 결과 정체성에
  없으면 GPU 서버에서 재실행할 때 resume이 CPU 결과를 "이미 있음"으로 건너뛴다.
- 결과 JSON만 보고 "어떤 하드웨어/라이브러리 버전에서 나온 숫자인가"를 알 수
  있어야 리더보드 수치를 감사할 수 있다.

장치 규칙:
- torch를 쓰지 않는 모델(uses_torch=False)은 항상 "cpu" — GPU 머신에서 돌려도
  Gen1-2 결과의 정체성(해시)이 바뀌지 않는다.
- torch 모델의 "auto"는 CUDA가 있으면 "cuda", 없으면 "cpu"로 해석된다.
- 해석된 장치는 config에 기록되고, "cpu"가 아닐 때만 config_hash에 들어간다
  (runner.results.HASH_EXEMPT_DEFAULTS — 기존 CPU 결과의 해시 하위호환).
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from importlib import metadata

VALID_DEVICES = ("auto", "cpu", "cuda")


def _cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def resolve_device(requested: str | None, uses_torch: bool) -> str:
    """config의 device 요청을 실제 실행 장치("cpu" | "cuda" | "cuda:N")로 해석한다."""
    requested = (requested or "auto").lower()
    if not uses_torch:
        return "cpu"
    if requested == "auto":
        return "cuda" if _cuda_available() else "cpu"
    if requested == "cpu":
        return "cpu"
    if requested.startswith("cuda"):
        if not _cuda_available():
            raise RuntimeError(f"device='{requested}' requested but CUDA is not available")
        return requested
    raise ValueError(f"unknown device '{requested}' (expected one of {VALID_DEVICES} or cuda:N)")


def _version(pkg: str) -> str | None:
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return None


def git_state() -> dict:
    """커밋 해시 + 워킹트리 dirty 여부 (dirty 결과는 커밋 해시만으로 재현 불가)."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        # 결과 파일 자체의 변경은 dirty로 치지 않는다 (벤치마크 실행 중 항상 바뀐다)
        changed = [
            ln
            for ln in status.splitlines()
            if ln.strip() and not ln[3:].startswith(("benchmarks/results/", "docs/assets/"))
        ]
        return {"commit": commit, "dirty": bool(changed)}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"commit": "unknown", "dirty": None}


def environment_fingerprint(device: str) -> dict:
    """결과 JSON에 남길 실행 환경 요약. 호스트명·사용자명 등 식별 정보는 넣지 않는다."""
    env: dict = {
        "device": device,
        "python": platform.python_version(),
        "platform": f"{platform.system()}-{platform.machine()}",
        "cpu_count": os.cpu_count(),
        "packages": {
            p: _version(p)
            for p in ("numpy", "scipy", "scikit-learn", "pandas", "stumpy", "torch")
            if _version(p) is not None
        },
    }
    torch = sys.modules.get("torch")
    if torch is not None:
        env["torch_cuda"] = getattr(torch.version, "cuda", None)
        if device.startswith("cuda") and torch.cuda.is_available():
            idx = torch.device(device).index or 0
            env["gpu"] = torch.cuda.get_device_name(idx)
            env["cudnn"] = torch.backends.cudnn.version()
            env["cudnn_deterministic"] = bool(torch.backends.cudnn.deterministic)
    return env
