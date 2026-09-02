"""TimesFM 어댑터 (Das et al., ICML 2024 / TimesFM 3.0, 2026).

두 세대의 체크포인트를 각각 다른 등록명으로 노출한다 — pip 패키지 안에서 API가
완전히 다르기 때문이다 (`timesfm` 모듈 = 2.5, `timesfm3` 모듈 = 3.0).

| 등록명        | 체크포인트                        | API                       | 특성                       |
|---------------|-----------------------------------|---------------------------|----------------------------|
| `timesfm`     | google/timesfm-2.5-200m-pytorch   | `timesfm.TimesFM_2p5_*`   | 단변량, 채널 독립          |
| `timesfm3`    | google/timesfm-3.0-pytorch        | `timesfm3.TimesFM3Evaluator` | 네이티브 다변량 (변량 어텐션) |
| `timesfm3_ci` | google/timesfm-3.0-pytorch        | 위와 동일 (`univariate=True`) | 채널 독립 대조군           |
| `timesfm3_prob` | google/timesfm-3.0-pytorch      | 위와 동일                  | 확률(분위) 점수 규약       |

`timesfm3` vs `timesfm3_ci`는 "네이티브 다변량이 이상탐지에도 도움이 되는가"를,
`timesfm3` vs `timesfm3_prob`는 "점 예측 대신 확률 예측을 점수로 쓰면 나은가"를
같은 가중치·같은 프로토콜로 분리 측정하기 위한 짝이다 (MambaTSAD faithful/fixed와
같은 취지의 ablation).

라이선스 (CLAUDE.md §10.2, THIRD_PARTY_NOTICES.md):
    코드는 pip 의존성으로만 연결한다(복사 없음). timesfm 패키지 코드는 Apache-2.0,
    2.5까지의 가중치도 Apache-2.0이지만, **TimesFM 3.0 가중치는
    `timesfm-non-commercial-license-v1.0`** 으로 비상업·비프로덕션 용도로 제한된다.
    3.0 어댑터는 fit()에서 이 사실을 UserWarning으로 경고한다.

TimesFM 1.x/2.0 API(`timesfm.TimesFm` + `TimesFmHparams`)는 현재 PyPI 패키지에
존재하지 않는다(`pip install timesfm==1.3.0` 필요). 2.5 어댑터는 신 API를 먼저
시도하고 실패하면 구 API로 폴백한다.
"""

from __future__ import annotations

import warnings

import numpy as np

from tsad_forge.models.gen5_ssm_foundation.foundation import (
    _ForecastResidualBase,
    _require,
)
from tsad_forge.models.registry import register_model

#: TimesFM 3.0 기본 가중치의 라이선스 경고 (비상업·비프로덕션 제한)
NONCOMMERCIAL_WARNING = (
    "TimesFM 3.0 사전학습 가중치는 timesfm-non-commercial-license-v1.0 하에 배포되어 "
    "비상업·비프로덕션 용도로만 사용할 수 있습니다 (패키지 코드는 Apache-2.0). "
    "상업/프로덕션 환경에서는 Apache-2.0인 2.5 가중치(model='timesfm')를 쓰세요. "
    "출처: https://github.com/google-research/timesfm"
)

#: TimesFM 3.0의 변량 어텐션 1회 forward 상한 (초과 시 라이브러리가 청크로 분할)
MAX_VARIATES_PER_FORWARD = 32


def _group_by_length(contexts: list[np.ndarray]) -> dict[int, list[int]]:
    """길이가 같은 컨텍스트끼리 묶는다 (배치 API가 동일 길이를 요구하는 경우 대비)."""
    groups: dict[int, list[int]] = {}
    for i, c in enumerate(contexts):
        groups.setdefault(len(c), []).append(i)
    return groups


@register_model("timesfm")
class TimesFMResidualDetector(_ForecastResidualBase):
    """TimesFM 2.5 (200M, Apache-2.0 가중치) 예측 잔차 — 채널 독립.

    2.5는 단변량 모델이므로 다변량 입력은 채널별로 독립 예측한 뒤 잔차를 평균한다.
    """

    supports_quantiles = True
    default_context = 512
    default_horizon = 32

    def __init__(
        self,
        seed: int = 0,
        model_name: str = "google/timesfm-2.5-200m-pytorch",
        normalize_inputs: bool = True,
        **params,
    ):
        super().__init__(
            seed=seed, model_name=model_name, normalize_inputs=normalize_inputs, **params
        )
        self.model_name = model_name
        self.normalize_inputs = bool(normalize_inputs)

    def _load(self) -> None:
        timesfm = _require("timesfm", "timesfm[torch]")
        if hasattr(timesfm, "TimesFM_2p5_200M_torch"):  # 현행 API (>=2.0 패키지)
            self.api_ = "2p5"
            model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(self.model_name)
            model.compile(
                timesfm.ForecastConfig(
                    max_context=self.context,
                    max_horizon=self.horizon,
                    normalize_inputs=self.normalize_inputs,
                    per_core_batch_size=self.batch_size,
                    use_continuous_quantile_head=True,
                    # z-score 정규화된 입력은 음수를 가지므로 비음수 강제를 끈다
                    infer_is_positive=False,
                    fix_quantile_crossing=True,
                )
            )
            self.model_ = model
        elif hasattr(timesfm, "TimesFm"):  # 레거시 1.x/2.0 API
            self.api_ = "v1"
            self.model_ = timesfm.TimesFm(
                hparams=timesfm.TimesFmHparams(context_len=self.context, horizon_len=self.horizon),
                checkpoint=timesfm.TimesFmCheckpoint(huggingface_repo_id=self.model_name),
            )
        else:  # pragma: no cover - 미래 버전 방어
            raise RuntimeError(
                "설치된 timesfm 패키지에서 TimesFM 2.5/1.x API를 찾지 못했습니다. "
                "`pip install timesfm[torch]` 로 재설치하거나 model='timesfm3'을 쓰세요."
            )

    def _forecast(self, context: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        inputs = [np.ascontiguousarray(context[:, d]) for d in range(context.shape[1])]
        if self.api_ == "2p5":
            point, quant = self.model_.forecast(horizon=self.horizon, inputs=inputs)
            point = np.asarray(point)[:, : self.horizon]  # [D, h]
            # 2.5 분위 출력은 [D, h, 10] = [평균, q0.1..q0.9]
            quant = np.asarray(quant)[:, : self.horizon, 1:]
            return point.T, np.transpose(quant, (1, 0, 2))
        fc, _ = self.model_.forecast(inputs, freq=[0] * len(inputs))
        return np.asarray(fc)[:, : self.horizon].T, None


class _TimesFM3Base(_ForecastResidualBase):
    """TimesFM 3.0 공통 구현 — 하위 클래스가 `univariate` 플래그만 바꾼다."""

    supports_quantiles = True
    default_context = 512
    default_horizon = 32
    univariate = False

    def __init__(
        self,
        seed: int = 0,
        model_name: str = "google/timesfm-3.0-pytorch",
        device: str | None = None,
        use_symmetric_averaging: bool = True,
        **params,
    ):
        super().__init__(
            seed=seed,
            model_name=model_name,
            device=device,
            use_symmetric_averaging=use_symmetric_averaging,
            **params,
        )
        self.model_name = model_name
        self.device = device
        self.use_symmetric_averaging = bool(use_symmetric_averaging)

    def _load(self) -> None:
        timesfm3 = _require("timesfm3", "timesfm[torch]")
        if self.model_name == "google/timesfm-3.0-pytorch":
            warnings.warn(NONCOMMERCIAL_WARNING, UserWarning, stacklevel=2)
        config = timesfm3.ModelConfig(
            checkpoint_path=self.model_name,
            per_core_batch_size=self.batch_size,
            device=self.device,
        )
        self.model_ = timesfm3.TimesFM3Evaluator(config)

    def _predict_batch(self, contexts: list[np.ndarray]) -> list:
        """contexts: [ctx, D] 리스트 -> ForecastOutput 리스트 (변량 축 우선으로 전치)."""
        return list(
            self.model_.predict_batch(
                contexts=[np.ascontiguousarray(c.T, dtype=np.float32) for c in contexts],
                horizon=self.horizon,
                return_quantiles=True,
                use_symmetric_averaging=self.use_symmetric_averaging,
                # z-score 정규화된 입력은 음수를 가지므로 비음수 클램프를 반드시 끈다
                make_positive=False,
                sort_quantiles=True,
                univariate=self.univariate,
            )
        )

    def _forecast_many(
        self, contexts: list[np.ndarray]
    ) -> list[tuple[np.ndarray, np.ndarray | None]]:
        out: list[tuple[np.ndarray, np.ndarray | None]] = [None] * len(contexts)  # type: ignore[list-item]
        for _, idx in _group_by_length(contexts).items():
            batch = self._predict_batch([contexts[i] for i in idx])
            for j, o in zip(idx, batch, strict=True):
                # forecast: [V, h], quantiles: [V, h, Q] -> [h, V], [h, V, Q]
                point = np.atleast_2d(np.asarray(o.forecast))
                quant = np.asarray(o.quantiles) if o.quantiles is not None else None
                if quant is not None and quant.ndim == 2:  # 단변량 반환 [h, Q]
                    quant = quant[None]
                out[j] = (
                    point.T,
                    np.transpose(quant, (1, 0, 2)) if quant is not None else None,
                )
        return out

    def _forecast(self, context: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        return self._forecast_many([context])[0]


@register_model("timesfm3")
class TimesFM3Detector(_TimesFM3Base):
    """TimesFM 3.0 — 네이티브 다변량 결합 예측 잔차.

    전 채널을 하나의 컨텍스트 `[D, ctx]`로 넣어 변량 간 어텐션으로 결합 예측한다.
    D > 32이면 라이브러리가 32변량씩 청크로 나누므로, 변량 어텐션은 청크 내부에서만
    작동한다 (SMD D=38, SWaT D=51 등에서 해당 — 결과 해석 시 유의).
    """

    univariate = False


@register_model("timesfm3_ci")
class TimesFM3ChannelIndependentDetector(_TimesFM3Base):
    """TimesFM 3.0 — 채널 독립(univariate) 대조군.

    `timesfm3`와 같은 가중치·같은 컨텍스트/호라이즌을 쓰되 변량 어텐션을 끄므로,
    두 결과의 차이가 곧 "네이티브 다변량"의 이상탐지 기여분이다.
    """

    univariate = True


@register_model("timesfm3_prob")
class TimesFM3ProbabilisticDetector(_TimesFM3Base):
    """TimesFM 3.0 — 확률(분위) 예측 기반 점수.

    `timesfm3`가 중앙값 예측의 절대 잔차를 쓰는 반면, 이 변형은 9개 분위 예측의
    pinball 평균(CRPS 근사)을 점수로 쓴다. 예측 벤치마크에서 point와 probabilistic
    순위가 따로 매겨지듯, 탐지에서도 두 규약을 분리해 측정한다.

    별도 등록명을 둔 이유: 결과 집계가 (model, dataset, channel, seed)로 중복을
    제거하므로(CLAUDE.md §5), 같은 이름 아래 model_params만 다른 변형은 리더보드에서
    서로를 덮어쓴다.
    """

    univariate = False
    default_score_mode = "crps"
