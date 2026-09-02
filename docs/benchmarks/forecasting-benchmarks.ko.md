# 예측 벤치마크와 TSAD 벤치마크

> "TimesFM-3는 GIFT-Eval, fev-bench, TIME 세 벤치마크에서 point/probabilistic
> 모두 평균 rank 1위."

사실이다. 그리고 이것은 **이상탐지에 대한 주장이 아니다**. 이 문서는 세 벤치마크가
실제로 무엇을 재는지, 그 주장 중 어디까지가 TSAD-Forge로 넘어오는지, 그리고 그
질문을 가정이 아니라 숫자로 답할 수 있게 하기 위해 이 저장소에 무엇을 추가했는지를
기록한다.

## 세 벤치마크

| 벤치마크 | 규모 | 과제 | 순위 지표 | 라이선스 |
|---|---|---|---|---|
| [GIFT-Eval](https://github.com/SalesforceAIResearch/gift-eval) (Aksu et al., ICLR 2025) | 23개 데이터셋, 97개 구성, 약 144k 시계열, 1.77억 포인트, 7개 도메인, 10개 주기 | 점/확률 예측 | MASE, CRPS (순위 집계) | Apache-2.0 |
| [fev-bench](https://arxiv.org/abs/2509.26468) (Shchur et al., 2025) | 100개 과제, 7개 도메인, 그중 46개는 공변량 포함 | 공변량 포함 예측 | SQL, MASE, WQL, WAPE → 승률 + skill score (부트스트랩 CI) | Apache-2.0 (`fev` 라이브러리) |
| [TIME](https://arxiv.org/abs/2602.12147) (2026) | 신규 데이터셋 50개, 과제 98개 | 사전학습 누출을 배제한 엄격한 zero-shot 예측 | MASE/CRPS 계열, 패턴 수준 집계 | 상위 저장소 확인 필요 |

세 벤치마크의 공통점 세 가지가 여기서 전부 중요하다:

1. **과제가 예측이다.** 컨텍스트 윈도우가 주어지면 다음 *h* 스텝을 맞힌다.
   "이 구간이 이상이다"라는 개념 자체가 없다.
2. **이상 라벨이 없다.** 250여 개 과제 어디에도 VUS-PR·affiliation-F1·event-F1이
   요구하는 이벤트 수준 정답이 없다.
3. **지표가 "평균적으로 맞히기"를 보상한다** (MASE, WQL, CRPS). 반면 TSAD 지표는
   극심하게 불균형한 라벨에서 희귀 이벤트를 **분리**하는 능력을 본다.

## 왜 데이터셋으로 추가하지 않았나

"이 벤치마크들을 TSAD-Forge에 추가할 수 있나?"의 정직한 답은 **벤치마크로는
안 된다**이다. GIFT-Eval의 97개 구성을 넣으면 라벨 없는 1.77억 포인트가 생길 뿐,
탐지기를 채점할 대상이 없다. 라벨을 인위적으로 만드는 것(예: 잔차 임계값으로 라벨
생성)은 [Wu & Keogh (TKDE 2021)](https://arxiv.org/abs/2009.13807)이 경고한, 탐지기에
유리하게 편향된 합성 정답을 만드는 일이고 벤치마크의 정의를 조용히 바꿔버린다.

GIFT-Eval의 TSAD 대응물은 이미 존재하고 이 저장소에 이미 연결되어 있다 —
**TSB-AD** (Liu & Paparrizos, NeurIPS 2024), [datasets/tsb_ad.md](../datasets/tsb_ad.md).
"파운데이션 모델이 이기는가"는 거기서 판가름 난다.

## 예측 1위가 탐지 1위가 아닌 이유

예측 잔차 기반 탐지기는 "모델이 얼마나 못 맞혔는가"로 점수를 매긴다. 여기서
예측을 잘할수록 **탐지가 나빠지는** 실패 모드가 생긴다:

- **모델이 이상까지 예측해버린다.** 긴 컨텍스트를 본 모델은 레벨 시프트의 전반부를
  보고 그대로 외삽한다. 잔차 ≈ 0 → 이벤트가 보이지 않는다. 오히려 약한 예측기가
  옛 레벨을 계속 예측해 이를 잡아낸다.
- **컨텍스트 오염.** TSAD에서 컨텍스트 윈도우는 곧 test 데이터다. 이상 구간이
  컨텍스트에 들어가고 모델은 그것에 조건화된다.
- **날카로움(sharpness)의 양면성.** 확률 지표(WQL/CRPS)는 잘 보정된 구간을
  보상한다. 넓지만 잘 보정된 구간은 예측 점수는 좋고, 구간 정규화 이상 점수는
  둔감해진다.
- **평균 1위 ≠ 도메인별 1위.** 집계 순위는 분포 이동을 가린다. 산업 센서
  데이터(SMD, SWaT, SKAB)는 예측 코퍼스를 지배하는 소매·에너지 계열과 전혀 다르다.

이 중 어느 것도 "TimesFM-3가 TSAD에서 나쁘다"는 말이 아니다. **아직 검증되지
않았다**는 말이고, 그 검증이 이 저장소의 일이다.

## 대신 무엇을 추가했나

아래는 전부 이 저장소에 있고 재현 가능하다.

### 1. 모델 — 세 가지 변형

| 등록명 | 체크포인트 | 분리해서 보는 것 |
|---|---|---|
| `timesfm3` | `google/timesfm-3.0-pytorch` | 네이티브 **다변량** 결합 예측(변량 어텐션), 점 잔차 점수 |
| `timesfm3_ci` | 동일 가중치 | **채널 독립** 대조군 — 같은 가중치, 변량 어텐션만 끔 |
| `timesfm3_prob` | 동일 가중치 | **확률** 점수: 9개 분위의 pinball 평균 (CRPS 근사) |
| `timesfm` | `google/timesfm-2.5-200m-pytorch` | TimesFM 2.5 기준선 (Apache-2.0 가중치) |

`timesfm3` vs `timesfm3_ci`는 3.0 릴리스의 대표 기능을 정확히 분리한다.
`timesfm3` vs `timesfm3_prob`는 예측 리더보드가 나누는 point/probabilistic 축을
탐지에서도 같은 방식으로 나눈다.

### 2. 탐지 지표 옆에 기록되는 예측 품질 지표

예측 기반 Gen5 실행은 이제 세 벤치마크가 순위를 매기는 지표를 같은 TSAD test
분할에서 함께 기록한다 (`tsad_forge/evaluation/forecast_metrics.py`):

- `fc_mase` — train 계절 나이브 MAE로 정규화한 MAE
- `fc_wql` — weighted quantile loss
- `fc_crps` — 분위 근사 CRPS
- `fc_mae` — 원 MAE

이 값들은 `benchmarks/results/*.parquet`에 `vus_pr` 바로 옆에 저장되므로
"예측 실력 vs 탐지 실력"은 새 실험이 아니라 조인 한 번이다. **`fc_*`가 낮은데
`vus_pr`도 낮은 칸이 핵심**이다 — 모델이 이상까지 잘 예측했다는 뜻이고, 잔차 기반
탐지기가 가장 감당할 수 없는 상황이다.

리더보드가 아니라 진단으로 읽어야 한다: 이상 구간이 포함된 분할에서 계산되므로
"낮을수록 좋다"는 예측에 대해서만 성립하고 탐지에는 아무 함의가 없다.

### 3. 실행 가능한 프로파일

```bash
pip install "tsad-forge[foundation]" "timesfm[torch]" chronos-forecasting momentfm
python benchmarks/run_all.py --profile configs/foundation.yaml
```

이 프로파일은 TimesFM 3.0/2.5·Chronos·MOMENT를 `lite` 프로파일과 **같은 엔티티**에
붙이므로 Gen1–Gen4 리더보드 행과 직접 비교된다. CI에서는 의도적으로 제외했고
(CLAUDE.md §9: CI에서 무거운 학습 금지) 가중치 다운로드에 HuggingFace 접근이 필요하다.

## 결과 현황

**이 저장소에서 아직 실행되지 않았다.** 사전학습 가중치는 HuggingFace 접근이
필요한데 어댑터를 작성한 환경에서는 접근이 차단되어 있어, `benchmarks/results/`에
TimesFM 행이 없고 리더보드에도 항목이 없다. 실행하지 않은 숫자를 적는 것은
CLAUDE.md §9 위반이다. 어댑터는 스텁 백엔드 기반 테스트(`tests/test_timesfm.py`)로
배열 규약·점수 모드·커버리지가 고정되어 있으므로, 모델 접근이 가능한 머신에서
프로파일을 실행하기만 하면 된다.

## 라이선스

`timesfm` 패키지 코드는 Apache-2.0이고 **pip 의존성으로만** 사용한다 —
코드 복사 없음 (CLAUDE.md §10.2).

**TimesFM 3.0 사전학습 가중치는 Apache-2.0이 아니다.**
`timesfm-non-commercial-license-v1.0`으로 배포되며 비상업·비프로덕션 용도로
제한된다. 2.5까지의 가중치는 Apache-2.0이다.

`timesfm3`·`timesfm3_ci`·`timesfm3_prob` 어댑터는 기본 3.0 체크포인트를 쓸 때
`fit()`에서 이 라이선스를 명시한 `UserWarning`을 낸다. 상업/프로덕션 평가에는
`timesfm`(2.5)를 쓰라. 가중치도 서드파티 코드도 재배포하지 않으므로 TSAD-Forge
자체의 Apache-2.0 라이선스에는 영향이 없다.
[THIRD_PARTY_NOTICES.md](https://github.com/Denny-Hwang/TSAD-Forge/blob/main/THIRD_PARTY_NOTICES.md) 참조.

## 참고문헌

- Das et al., *A decoder-only foundation model for time-series forecasting*, ICML 2024. [arXiv:2310.10688](https://arxiv.org/abs/2310.10688)
- TimesFM 3.0 릴리스 및 라이선스 고지 — [google-research/timesfm](https://github.com/google-research/timesfm)
- Aksu et al., *GIFT-Eval: A Benchmark For General Time Series Forecasting Model Evaluation*, ICLR 2025. [arXiv:2410.10393](https://arxiv.org/abs/2410.10393)
- Shchur et al., *fev-bench: A Realistic Benchmark for Time Series Forecasting*, 2025. [arXiv:2509.26468](https://arxiv.org/abs/2509.26468)
- *It's TIME: Towards the Next Generation of Time Series Forecasting Benchmarks*, 2026. [arXiv:2602.12147](https://arxiv.org/abs/2602.12147)
- Liu & Paparrizos, *TSB-AD*, NeurIPS 2024 — TSAD 대응 벤치마크.
- Wu & Keogh, *Current Time Series Anomaly Detection Benchmarks are Flawed*, TKDE 2021. [arXiv:2009.13807](https://arxiv.org/abs/2009.13807)
