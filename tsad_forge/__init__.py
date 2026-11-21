"""TSAD-Forge: a reproducible benchmarking and learning ecosystem for time-series anomaly detection."""

from importlib.metadata import PackageNotFoundError, version

try:  # pyproject.toml의 버전이 단일 출처 (하드코딩하면 릴리스 때마다 어긋난다)
    __version__ = version("tsad-forge")
except PackageNotFoundError:  # 설치하지 않고 소스 트리에서 import한 경우
    __version__ = "0+unknown"
