# ch02 — Gen1 통계적 방법 (1930s–2000s)

> 대응 노트북: `notebooks/ch02_gen1_statistical.ipynb` — PCA-T²/SPE를 SMD에 적용

## 관리도 (Control Charts)

산업 품질관리에서 출발한 최초의 "이상탐지"다.

- **CUSUM** (Page 1954): 표준화 편차의 누적합 \(S^+_t = \max(0, S^+_{t-1} + z_t - k)\).
  작은 지속적 평균 이동(mean shift)에 민감. slack \(k\)는 탐지 대상 이동폭의 절반.
- **EWMA** (Roberts 1959): \(z_t = \lambda x_t + (1-\lambda) z_{t-1}\).
  최근 관측 가중 — CUSUM보다 급격한 변화에 반응이 빠르다.

구현: `tsad_forge/models/gen1_statistical/control_charts.py`

## Hotelling T²와 PCA-T²/SPE 유도

다변량 정상 데이터 \(x \sim \mathcal{N}(\mu, \Sigma)\)에서 마할라노비스 거리
\(T^2 = (x-\mu)^\top \Sigma^{-1} (x-\mu)\) 는 자유도 D의 \(\chi^2\) 분포를 따른다.

D가 크면 \(\Sigma^{-1}\)이 불안정하므로 PCA로 차원을 나눈다:

- **T² (주성분 공간)**: 상위 k개 주성분 점수의 정규화 제곱합 —
  "정상 변동 방향 안에서 과도하게 큰 움직임"
- **SPE/Q (잔차 공간)**: \(\|x - \hat{x}\|^2\) (재구성 오차) —
  "정상 변동 방향 밖으로 벗어남"

두 통계량은 **서로 다른 종류의 이상**을 잡는다. 반도체 FDC에서 T²는 공정 드리프트,
SPE는 센서 고장·신규 fault 모드에 대응하는 경우가 많다 (ch09).

## Sub-PCA — 얕보면 안 되는 baseline

시계열을 슬라이딩 윈도우로 잘라 PCA 재구성 오차를 점수로 쓰는 단순한 방법이
TSB-AD 리더보드 상위권이다. 시간 구조를 윈도우로 포착하면서 선형 모델의
안정성을 유지하기 때문이다. 이 저장소 리더보드에서 직접 확인하라.

## STL 잔차

STL(Season-Trend decomposition using Loess)로 추세·계절 성분을 제거한 잔차의 |z|를
점수로 사용. 주기 추정은 ACF 최대 피크로 자동화했다 (`estimate_period`).
주기가 없는 시계열은 이동평균 detrend로 폴백한다.
