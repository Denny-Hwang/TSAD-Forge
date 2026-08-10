"""모델 레지스트리: 이름 -> BaseDetector 서브클래스."""

from __future__ import annotations

from tsad_forge.models.base import BaseDetector

_REGISTRY: dict[str, type[BaseDetector]] = {}


def register_model(name: str):
    """데코레이터: @register_model("iforest")"""

    def deco(cls: type[BaseDetector]) -> type[BaseDetector]:
        if name in _REGISTRY:
            raise ValueError(f"model '{name}' already registered")
        if not issubclass(cls, BaseDetector):
            raise TypeError(f"{cls} must subclass BaseDetector")
        _REGISTRY[name] = cls
        return cls

    return deco


def get_model(name: str, **kwargs) -> BaseDetector:
    _ensure_builtin()
    if name not in _REGISTRY:
        raise KeyError(f"unknown model '{name}'. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)


def get_model_class(name: str) -> type[BaseDetector]:
    _ensure_builtin()
    if name not in _REGISTRY:
        raise KeyError(f"unknown model '{name}'. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def list_models() -> list[str]:
    _ensure_builtin()
    return sorted(_REGISTRY)


_BUILTIN_LOADED = False


def _ensure_builtin() -> None:
    """내장 모델 등록 (지연 import로 순환 방지)."""
    global _BUILTIN_LOADED
    if _BUILTIN_LOADED:
        return
    _BUILTIN_LOADED = True
    import tsad_forge.models.dummy  # noqa: F401  (등록 부수효과)
