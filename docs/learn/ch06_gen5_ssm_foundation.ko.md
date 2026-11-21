# ch06 — Gen5 SSM/Mamba와 파운데이션 모델 (2023–)

> 대응 노트북: `notebooks/ch06_gen5.ipynb` — MambaTSAD faithful vs fixed (v2) 비교

## 상태공간 모델(SSM)과 선택적 게이팅

Mamba(S6)의 핵심은 상태 갱신 \(h_t = \bar{A}_t h_{t-1} + \bar{B}_t x_t\)의 파라미터를
**입력에 따라 바꾸는 선택성(selectivity)**이다: \(\Delta_t, B_t, C_t = f(x_t)\).
Transformer의 O(T²) 어텐션 없이 긴 의존성을 O(T)로 처리한다.

**Δ 이산화와 균일 샘플링 가정**: 연속 SSM을 \(\bar{A} = \exp(\Delta A)\)로 이산화할 때
Δ는 "샘플 간 시간 간격"의 역할을 한다. 즉 **샘플링이 균일하다는 가정**이 깔려 있다 —
불규칙 샘플링 데이터(ch09의 산업 현장)에 SSM을 그대로 쓰면 이 가정이 깨진다.

## MambaTSAD와 faithful/fixed 실험

> **정정 (v2).** 이 페이지의 이전 판은 "알려진 이슈 4가지"와 faithful 대 fixed 결과표
> (평균 VUS-PR 0.251 → 0.502)를 실었다. 그 v1 구현은 **공식 코드를 확인하지 않고**
> 작성됐고, 이슈 대부분이 공식 동작과 달랐다. 예를 들어 공식 모델은 trend가 아니라
> HP *순환* 성분을 입력으로 쓰고, 선택성을 끄는 CPU 분기는 존재하지 않는다. 해당
> 결과는 철회해 `benchmarks/results/_superseded/mamba_tsad_v1/`로 격리했으며
> 인용하면 안 된다. 교훈은 우리 자신에게 그대로 적용된다: **재현을 주장하기 전에
> 원 코드를 확인하라.**

v2는 공식 저장소(aaaceo890/mamba_tsad @ `08fec05`)를 읽은 뒤 다시 만든 동작
재구현이다. 저장소에 라이선스가 없어 코드는 복사하지 않았다.

- **모델.** HP 사전 필터 → residual 없는 RMSNorm→S6 블록 3개, 각 블록 뒤
  FFT 주기 이동평균(AMA) 분해.
- **학습.** 셔플 없는 순차 배치, 배치 간 hidden state 전달.
- **점수.** 겹치지 않는 test 윈도우.

공식 코드에서 확인한 이슈:

| 이슈 | 공식 동작 | 플래그 (faithful → fixed) |
|---|---|---|
| (a) 추론 hidden state 인덱싱 | 다음 test 윈도우로 넘기는 상태가 `hs[:, b-1]`로, 시간축을 배치 크기로 인덱싱한다. test 배치 1이면 마지막이 아닌 **첫 시점** 상태가 넘어간다 | `state_index: upstream → last` |
| (b) CPU/GPU 분기 | 학습 루프의 CPU 분기도 `.cuda()`를 호출해 공식 코드는 CPU에서 학습할 수 없다 (공식 결과는 GPU 전용) | 없음 (이 구현은 장치 무관) |
| (c) HP 목적함수 | HP 정의 `(I + λD₂ᵀD₂)τ = y` 대신 1차 차분 항이 추가된 `(I + 2λD₁ᵀD₁ + λD₂ᵀD₂)τ = y`를 푼다 | `hp_objective: upstream → standard` |
| (d) AMA 주기 | PSD를 **배치 전체**와 모든 채널에서 평균해 배치당 주기 1개를 쓴다. 그래서 학습(배치 128)과 추론(배치 1)이 서로 다른 모집단에서 주기를 추정한다 | `ama_period: batch → sample` |

재현:

- `configs/mamba_paper_repro.yaml`: 논문 엔티티 + 공식 하이퍼파라미터.
- `configs/mamba_ablation.yaml`: faithful, 이슈 하나씩만 수정한 변형, 전부 수정한 변형을 `model_alias`로 리더보드의 별도 행으로 실행.
- `configs/mamba_compare.yaml`: 소형 CPU 스모크.

v2 결과는 아직 게시하지 않았다 — 공식 프로토콜은 GPU가 필요하다 (CPU에서는 SMD
머신 1개에 epoch당 약 1시간).

## 파운데이션 모델과 zero-shot

MOMENT·Chronos·TimesFM은 대규모 시계열 코퍼스로 사전학습된 범용 모델이다.
TSAD 어댑터 두 방식:

- **zero-shot 재구성** (`moment`): 마스킹 재구성 오차를 그대로 점수로
- **예측 잔차** (`chronos`, `timesfm`, `timesfm3`): 예측 모델의 잔차를 점수로

학습 데이터가 전혀 없는 cold start(ch09)에서 매력적이지만, (1) 추론 비용이 크고
(2) 사전학습 분포와 다른 도메인(산업 센서)에서 성능이 급락할 수 있다.
어댑터는 `pip install tsad-forge[foundation]` 후 사용한다.

### 예측 잔차 공통 프로토콜

예측 기반 어댑터는 서로 비교 가능하도록 하나의 프로토콜을 고정한다. `fit()`은
zero-shot이고(가중치 로드 + train 꼬리 `context` 스텝을 초기 컨텍스트로 보관),
`score()`는 stride = `horizon`으로 슬라이딩하며 **test의 모든 스텝을 정확히 한 번씩**
예측한다. 점수 규약은 세 가지다: `residual`(|x − 중앙값|, 기본값이자 리더보드에
쓰는 규약), `crps`(분위 pinball 평균), `interval`(예측 구간 폭으로 정규화한 이탈도).

### TimesFM 3.0과 "rank 1위" 질문

TimesFM 3.0(2026년 8월, 330M 파라미터, 약 1조 포인트 학습)은 변량 어텐션 기반의
네이티브 다변량 예측을 추가했고, GIFT-Eval·fev-bench·TIME에서 point와 probabilistic
모두 평균 rank 1위다. 그러나 이들은 **예측** 벤치마크이고 이상 라벨이 없으므로,
그 순위가 TSAD에 대해 직접 말해주는 것은 없다. 이 저장소는 같은 가중치·같은 설정의
세 변형으로 그 주장을 측정 가능한 질문으로 바꿔놓았다:

| 등록명 | 분리해서 보는 것 |
|---|---|
| `timesfm3` | 네이티브 다변량 결합 예측, 점 잔차 |
| `timesfm3_ci` | 채널 독립 대조군 (변량 어텐션 off) |
| `timesfm3_prob` | 확률 점수 (9개 분위의 CRPS 근사) |

모든 실행은 `vus_pr` 옆에 `fc_mase`/`fc_wql`/`fc_crps`를 함께 기록하므로, 예측
실력과 탐지 실력을 같은 행에서 읽을 수 있다. 여기서 직관에 반하는 지점이 핵심이다:
**예측을 더 잘하는 모델이 더 나쁜 탐지기가 될 수 있다.** 컨텍스트가 충분한 모델은
레벨 시프트를 그대로 외삽해버려 잔차를 거의 0으로 만들기 때문이다.

읽을 때 유의할 점: TimesFM 3.0의 변량 어텐션은 1회 forward당 최대 32채널까지만
적용된다(SMD는 38, SWaT는 51 — 라이브러리가 청크로 분할한다). 그리고 3.0 가중치는
비상업 라이선스이고 2.5는 Apache-2.0이다. 전체 논의:
[예측 벤치마크와 TSAD 벤치마크](../benchmarks/forecasting-benchmarks.ko.md).
