# ch06 — Gen5 SSM/Mamba와 파운데이션 모델 (2023–)

> 대응 노트북: `notebooks/ch06_gen5.ipynb` — MambaTSAD faithful vs fixed 비교

## 상태공간 모델(SSM)과 선택적 게이팅

Mamba(S6)의 핵심은 상태 갱신 \(h_t = \bar{A}_t h_{t-1} + \bar{B}_t x_t\)의 파라미터를
**입력에 따라 바꾸는 선택성(selectivity)**이다: \(\Delta_t, B_t, C_t = f(x_t)\).
Transformer의 O(T²) 어텐션 없이 긴 의존성을 O(T)로 처리한다.

**Δ 이산화와 균일 샘플링 가정**: 연속 SSM을 \(\bar{A} = \exp(\Delta A)\)로 이산화할 때
Δ는 "샘플 간 시간 간격"의 역할을 한다. 즉 **샘플링이 균일하다는 가정**이 깔려 있다 —
불규칙 샘플링 데이터(ch09의 산업 현장)에 SSM을 그대로 쓰면 이 가정이 깨진다.

## MambaTSAD와 faithful/fixed 실험

MambaTSAD (Chen et al., IEEE SPL 2024)의 공개 구현에는 알려진 이슈 4가지가 있다.
이 저장소는 두 변형을 나란히 제공해 **구현 품질이 벤치마크 수치에 미치는 영향**을
정량화한다 (`mamba_tsad_faithful` vs `mamba_tsad_fixed`):

| 이슈 | 내용 |
|---|---|
| hidden state 인덱싱 | 출력이 \(h_{t-1}\)을 참조하는 off-by-one |
| CPU/GPU 분기 | CPU 경로에서 선택성 상실 → 장치에 따라 다른 모델이 됨 |
| HP filter 목적 | 순환(cycle) 대신 추세(trend)를 입력으로 사용 |
| AMA 전역 FFT | 전체 시계열 FFT 1회로 고정 창 → 국소 주기 변화 무시 |

재현 커맨드: `python benchmarks/run_all.py --profile configs/mamba_compare.yaml`
결과는 리더보드에서 두 모델로 분리 표시된다. **교훈: 논문 성능의 상당 부분이
구현 세부에 좌우될 수 있다 — 재현 연구가 필요한 이유다.**

## 파운데이션 모델과 zero-shot

MOMENT·Chronos·TimesFM은 대규모 시계열 코퍼스로 사전학습된 범용 모델이다.
TSAD 어댑터 두 방식:

- **zero-shot 재구성** (`moment`): 마스킹 재구성 오차를 그대로 점수로
- **예측 잔차** (`chronos`, `timesfm`): 예측 모델의 잔차를 점수로

학습 데이터가 전혀 없는 cold start(ch09)에서 매력적이지만, (1) 추론 비용이 크고
(2) 사전학습 분포와 다른 도메인(산업 센서)에서 성능이 급락할 수 있다.
어댑터는 `pip install tsad-forge[foundation]` 후 사용한다.
