# ch05 — Gen4 그래프와 어텐션 (2020–2023)

> 대응 노트북: `notebooks/ch05_gen4.ipynb`

## 그래프: 센서 간 관계의 명시적 모델링 — GDN

다변량 시계열에서 "어떤 센서가 어떤 센서와 함께 움직이는가"는 이상탐지의 핵심 정보다.
GDN (Deng & Hooi, AAAI 2021):

1. 센서별 **학습 가능한 임베딩** \(v_i\)
2. 임베딩 코사인 유사도 top-k로 **그래프 구성**
3. 그래프 어텐션으로 이웃 정보를 모아 **다음 값 예측**
4. 센서별 예측 편차의 강건 정규화 최대값이 점수 → **어느 센서가 원인인지** 해석 가능

## 어텐션: Association Discrepancy — Anomaly Transformer

Anomaly Transformer (Xu et al., ICLR 2022)의 통찰: **이상 지점의 어텐션은 자기 주변에만
집중된다** (전역 패턴과 연관을 만들지 못함).

- series-association: 데이터에서 학습된 어텐션 분포
- prior-association: 거리 기반 가우시안 (국소 집중의 기준선)
- 두 분포의 KL 거리(association discrepancy)가 **작을수록** 이상 —
  재구성 오차와 곱해 점수를 만든다

DCdetector (KDD 2023)는 같은 직관을 **재구성 없이** 이중 어텐션 뷰(patch-wise vs
in-patch)의 대조 학습만으로 구현했다.

## TimesNet: 주기의 2D 재배열

FFT로 지배 주기를 찾아 1D 시계열을 (주기 × 주기 내 위치)의 2D로 접은 뒤 2D conv를
적용한다. 주기 내 패턴(intraperiod)과 주기 간 변화(interperiod)를 동시에 본다.

## 정직한 질문

이 세대는 벤치마크 부풀림 논란의 중심이기도 하다. Anomaly Transformer의 원 보고는
PA-F1 기준이었고, PA 없이 재평가하면 순위가 크게 바뀐다 (Kim et al., AAAI 2022;
TSB-AD, NeurIPS 2024). 이 저장소의 [리더보드](../leaderboard/index.md)에서
VUS-PR 기준 실제 순위를 확인하라 — Gen1–2 baseline과의 비교가 핵심이다.
