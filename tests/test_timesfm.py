"""TimesFM 어댑터 테스트 (2.5 / 3.0) + 예측 잔차 공통 프로토콜.

사전학습 가중치는 HuggingFace 접근이 필요하므로 CI에서 받을 수 없다. 대신 실제
라이브러리와 **같은 형태의 반환값**을 내는 가짜 백엔드를 sys.modules에 주입해
어댑터의 배열 변환(변량 축 전치, 분위 축 정렬)·커버리지·점수 규약을 검증한다.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

# 어댑터 자체는 torch가 필요 없지만(백엔드가 자체 의존성을 가짐), gen5 패키지 __init__이
# mamba_tsad를 import하므로 registry 경로가 torch에 묶여 있다. torch 없는 CI 잡에서는
# 수집 단계에서 죽지 않도록 건너뛴다 (test_gen5_models.py와 같은 규약, CI test-dl 잡이 실행).
pytest.importorskip("torch", reason="Gen5 등록 경로가 torch 필요 (extras: dl)")

from tsad_forge.evaluation.forecast_metrics import DEFAULT_QUANTILE_LEVELS  # noqa: E402
from tsad_forge.models.gen5_ssm_foundation.foundation import (  # noqa: E402
    _ForecastResidualBase,
)
from tsad_forge.models.gen5_ssm_foundation.timesfm import (  # noqa: E402
    NONCOMMERCIAL_WARNING,
)
from tsad_forge.models.registry import get_model, list_models  # noqa: E402

RNG = np.random.default_rng(3)
T_TR, T_TE, D = 300, 160, 3
_t = np.arange(T_TR + T_TE)
_base = np.sin(2 * np.pi * _t / 40)[:, None] + RNG.normal(scale=0.05, size=(T_TR + T_TE, D))
TRAIN, TEST = _base[:T_TR], _base[T_TR:].copy()
ANOM = slice(60, 75)
TEST[ANOM] += 6.0

NQ = len(DEFAULT_QUANTILE_LEVELS)


# --- 가짜 백엔드 ------------------------------------------------------------


def _persistence(ctx_2d: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """마지막 값 유지 예측. ctx_2d [V, L] -> (point [V, h], quantiles [V, h, Q])."""
    point = np.repeat(ctx_2d[:, -1:], horizon, axis=1).astype(np.float64)
    spread = np.linspace(-1.0, 1.0, NQ)
    return point, point[..., None] + spread


class _FakeForecastOutput:
    def __init__(self, forecast, quantiles):
        self.ts_id = None
        self.forecast = forecast
        self.quantiles = quantiles


class _FakeTimesFM3Evaluator:
    """timesfm3.TimesFM3Evaluator와 같은 반환 규약을 흉내낸다."""

    last_kwargs: dict = {}

    def __init__(self, config):
        self.config = config

    def predict_batch(self, contexts, horizon, **kwargs):
        _FakeTimesFM3Evaluator.last_kwargs = dict(kwargs)
        for ctx in contexts:
            ctx = np.atleast_2d(np.asarray(ctx))
            point, quant = _persistence(ctx, horizon)
            if kwargs.get("univariate"):  # 변량 어텐션 없이 채널별로 푼 결과라고 가정
                point = point + 0.0
            yield _FakeForecastOutput(point, quant if kwargs.get("return_quantiles") else None)


class _FakeTimesFM2p5:
    def __init__(self):
        self.compiled = None

    @classmethod
    def from_pretrained(cls, name):
        obj = cls()
        obj.name = name
        return obj

    def compile(self, forecast_config):
        self.compiled = forecast_config

    def forecast(self, horizon, inputs):
        """2.5 규약: point [N, h], quantile [N, h, 10] (0번은 평균, 1..9가 분위)."""
        points, quants = [], []
        for series in inputs:
            p, q = _persistence(np.asarray(series)[None, :], horizon)
            points.append(p[0])
            quants.append(np.concatenate([p[0][:, None], q[0]], axis=-1))
        return np.stack(points), np.stack(quants)


@pytest.fixture
def fake_timesfm3(monkeypatch):
    mod = types.ModuleType("timesfm3")
    mod.ModelConfig = lambda **kw: kw  # type: ignore[attr-defined]
    mod.TimesFM3Evaluator = _FakeTimesFM3Evaluator  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "timesfm3", mod)
    return mod


@pytest.fixture
def fake_timesfm25(monkeypatch):
    mod = types.ModuleType("timesfm")
    mod.TimesFM_2p5_200M_torch = _FakeTimesFM2p5  # type: ignore[attr-defined]
    mod.ForecastConfig = lambda **kw: kw  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "timesfm", mod)
    return mod


SMALL = {"context": 48, "horizon": 8, "batch_size": 4}


# --- 등록 / 가드 ------------------------------------------------------------


def test_timesfm_models_registered():
    for name in ("timesfm", "timesfm3", "timesfm3_ci", "timesfm3_prob"):
        assert name in list_models()


def test_timesfm3_guarded_without_library(monkeypatch):
    monkeypatch.setitem(sys.modules, "timesfm3", None)
    with pytest.raises(RuntimeError, match="pip install"):
        get_model("timesfm3", **SMALL).fit(TRAIN)


def test_timesfm3_warns_about_noncommercial_weights(fake_timesfm3):
    with pytest.warns(UserWarning, match="non-commercial"):
        get_model("timesfm3", **SMALL).fit(TRAIN)


def test_custom_checkpoint_does_not_warn(fake_timesfm3):
    """비기본 체크포인트(예: 자체 파인튜닝 가중치)에는 경고를 붙이지 않는다."""
    with warnings_as_errors():
        get_model("timesfm3", model_name="/local/my-ckpt", **SMALL).fit(TRAIN)


class warnings_as_errors:
    def __enter__(self):
        import warnings

        self._cm = warnings.catch_warnings()
        self._cm.__enter__()
        warnings.simplefilter("error", UserWarning)
        return self

    def __exit__(self, *exc):
        return self._cm.__exit__(*exc)


def test_noncommercial_warning_names_the_license():
    assert "timesfm-non-commercial-license-v1.0" in NONCOMMERCIAL_WARNING


# --- 점수 규약 --------------------------------------------------------------


@pytest.mark.parametrize("name", ["timesfm3", "timesfm3_ci"])
def test_timesfm3_score_contract(fake_timesfm3, name):
    model = get_model(name, **SMALL).fit(TRAIN)
    scores = model.score(TEST)
    assert scores.shape == (T_TE,)
    assert np.isfinite(scores).all()
    assert model.generation == "gen5"
    # 이상 구간의 시작(레벨 시프트 발생 시점)이 정상 구간 상위값보다 확실히 높아야 한다.
    # 구간 전체 평균이 아니라 최댓값으로 보는 이유는 아래 회귀 테스트 참조.
    rest = np.delete(scores, np.arange(T_TE)[ANOM])
    assert scores[ANOM].max() > 5 * np.percentile(rest, 95)


def test_sustained_anomaly_is_absorbed_into_the_context(fake_timesfm3):
    """예측 잔차 탐지의 구조적 사각지대 — 문서화된 실패 모드의 회귀 테스트.

    지속되는 레벨 시프트는 곧 컨텍스트에 들어가고, 예측기는 그것을 그대로
    외삽한다. 잔차는 이벤트 시작에서만 크고 내부에서는 정상 수준으로 내려간다
    (docs/benchmarks/forecasting-benchmarks.md — "모델이 이상까지 예측해버린다").
    """
    scores = get_model("timesfm3", **SMALL).fit(TRAIN).score(TEST)
    onset = scores[ANOM.start : ANOM.start + 4]
    inside = scores[ANOM.start + 8 : ANOM.stop]
    assert onset.mean() > 10 * inside.mean()


def test_timesfm3_reports_forecast_diagnostics(fake_timesfm3):
    model = get_model("timesfm3", **SMALL).fit(TRAIN)
    model.score(TEST)
    assert set(model.diagnostics_) == {"fc_mase", "fc_mae", "fc_wql", "fc_crps"}
    assert all(np.isfinite(v) for v in model.diagnostics_.values())


@pytest.mark.parametrize("mode", ["residual", "crps", "interval"])
def test_score_modes_all_flag_the_anomaly(fake_timesfm3, mode):
    scores = get_model("timesfm3", score_mode=mode, **SMALL).fit(TRAIN).score(TEST)
    assert np.isfinite(scores).all()
    rest = np.delete(scores, np.arange(T_TE)[ANOM])
    assert scores[ANOM].mean() > rest.mean()


def test_unknown_score_mode_rejected():
    with pytest.raises(ValueError, match="score_mode"):
        get_model("timesfm3", score_mode="nope")


def test_quantile_mode_requires_quantiles(fake_timesfm3):
    """점 예측만 내는 백엔드에서 crps 모드를 쓰면 명확히 실패해야 한다."""
    model = get_model("timesfm3", score_mode="crps", **SMALL).fit(TRAIN)
    model._forecast_many = lambda ctxs: [  # type: ignore[method-assign]
        (np.zeros((model.horizon, TEST.shape[1])), None) for _ in ctxs
    ]
    with pytest.raises(RuntimeError, match="분위 예측이 필요"):
        model.score(TEST)


def test_evaluator_flags_are_safe_for_zscored_input(fake_timesfm3):
    """z-score 입력은 음수를 가지므로 make_positive는 반드시 꺼져 있어야 한다."""
    get_model("timesfm3", **SMALL).fit(TRAIN).score(TEST)
    kwargs = _FakeTimesFM3Evaluator.last_kwargs
    assert kwargs["make_positive"] is False
    assert kwargs["return_quantiles"] is True
    assert kwargs["sort_quantiles"] is True
    assert kwargs["univariate"] is False


def test_channel_independent_variant_sets_univariate(fake_timesfm3):
    get_model("timesfm3_ci", **SMALL).fit(TRAIN).score(TEST)
    assert _FakeTimesFM3Evaluator.last_kwargs["univariate"] is True


# --- 배열 변환 (전치 버그 회귀) ---------------------------------------------


def test_variate_axis_is_transposed_correctly(fake_timesfm3):
    """백엔드는 [V, h]로 주고 어댑터는 [h, V]로 써야 한다 — 채널이 섞이면 안 된다."""
    model = get_model("timesfm3", **SMALL).fit(TRAIN)
    ctx = np.zeros((SMALL["context"], 3))
    ctx[-1] = [1.0, 20.0, 300.0]  # 채널별로 확연히 다른 마지막 값
    point, quant = model._forecast(ctx)
    assert point.shape == (SMALL["horizon"], 3)
    assert quant is not None and quant.shape == (SMALL["horizon"], 3, NQ)
    np.testing.assert_allclose(point[0], [1.0, 20.0, 300.0])
    # 분위 축도 채널을 따라가야 한다 (중앙 분위 = 점 예측)
    np.testing.assert_allclose(quant[0, :, NQ // 2], [1.0, 20.0, 300.0])


def test_univariate_backend_output_is_accepted(fake_timesfm3):
    """단변량 입력에서 백엔드가 [h]/[h, Q]로 축을 줄여 반환해도 처리해야 한다."""

    class _Squeezing(_FakeTimesFM3Evaluator):
        def predict_batch(self, contexts, horizon, **kwargs):
            for out in super().predict_batch(contexts, horizon, **kwargs):
                yield _FakeForecastOutput(out.forecast[0], out.quantiles[0])

    fake_timesfm3.TimesFM3Evaluator = _Squeezing
    model = get_model("timesfm3", **SMALL).fit(TRAIN[:, :1])
    scores = model.score(TEST[:, :1])
    assert scores.shape == (T_TE,) and np.isfinite(scores).all()


# --- 공통 프로토콜 (커버리지) ------------------------------------------------


class _StubForecaster(_ForecastResidualBase):
    """예측 커버리지 검사를 위한 화이트박스 스텁."""

    def _load(self):
        self.captured_point = None

    def _forecast(self, context):
        point, quant = _persistence(np.asarray(context).T, self.horizon)
        return point.T, np.transpose(quant, (1, 0, 2))

    def _to_scores(self, X, point, quant):
        self.captured_point = point.copy()
        return super()._to_scores(X, point, quant)


def test_every_test_step_is_predicted_exactly_once():
    """train 꼬리를 초기 컨텍스트로 쓰므로 test 앞머리에도 빈 예측이 없어야 한다."""
    model = _StubForecaster(context=48, horizon=8).fit(TRAIN)
    model.score(TEST)
    assert np.isfinite(model.captured_point).all()


def test_short_train_still_covers_the_whole_test():
    model = _StubForecaster(context=48, horizon=8).fit(TRAIN[:10])
    model.score(TEST)
    assert np.isfinite(model.captured_point).all()


def test_horizon_not_dividing_test_length_is_covered():
    model = _StubForecaster(context=48, horizon=7).fit(TRAIN)
    model.score(TEST)
    assert np.isfinite(model.captured_point).all()


def test_context_and_horizon_are_configurable_and_validated():
    m = _StubForecaster(context=17, horizon=5)
    assert (m.context, m.horizon) == (17, 5)
    assert m.get_config()["context"] == 17  # 결과 JSON에 기록되는 재현 정보
    with pytest.raises(ValueError, match="context/horizon"):
        _StubForecaster(context=0)


# --- TimesFM 2.5 -------------------------------------------------------------


def test_timesfm25_adapter_uses_current_api(fake_timesfm25):
    model = get_model("timesfm", **SMALL).fit(TRAIN)
    assert model.api_ == "2p5"
    cfg = model.model_.compiled
    assert cfg["max_context"] == SMALL["context"] and cfg["max_horizon"] == SMALL["horizon"]
    assert cfg["infer_is_positive"] is False  # z-score 입력 방어
    scores = model.score(TEST)
    assert scores.shape == (T_TE,) and np.isfinite(scores).all()
    rest = np.delete(scores, np.arange(T_TE)[ANOM])
    assert scores[ANOM].max() > 5 * np.percentile(rest, 95)


def test_timesfm25_strips_the_mean_channel_from_quantiles(fake_timesfm25):
    """2.5 분위 출력의 0번 축은 평균이다 — 9개 분위만 남겨야 한다."""
    model = get_model("timesfm", **SMALL).fit(TRAIN)
    _, quant = model._forecast(np.zeros((SMALL["context"], 2)))
    assert quant is not None and quant.shape == (SMALL["horizon"], 2, NQ)


def test_timesfm_legacy_v1_api_fallback(monkeypatch):
    """구 API(TimesFm)만 있는 환경에서는 점 예측 폴백으로 동작한다."""

    class _LegacyTimesFm:
        def __init__(self, hparams, checkpoint):
            self.h = hparams

        def forecast(self, inputs, freq):
            return np.stack([np.repeat(np.asarray(s)[-1], 8) for s in inputs]), None

    mod = types.ModuleType("timesfm")
    mod.TimesFm = _LegacyTimesFm  # type: ignore[attr-defined]
    mod.TimesFmHparams = lambda **kw: kw  # type: ignore[attr-defined]
    mod.TimesFmCheckpoint = lambda **kw: kw  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "timesfm", mod)

    model = get_model("timesfm", **SMALL).fit(TRAIN)
    assert model.api_ == "v1"
    scores = model.score(TEST)
    assert scores.shape == (T_TE,) and np.isfinite(scores).all()
    assert "fc_wql" not in model.diagnostics_  # 점 예측만 → 확률 지표 없음


def test_unsupported_timesfm_package_raises(monkeypatch):
    monkeypatch.setitem(sys.modules, "timesfm", types.ModuleType("timesfm"))
    with pytest.raises(RuntimeError, match="TimesFM 2.5/1.x API"):
        get_model("timesfm", **SMALL).fit(TRAIN)
