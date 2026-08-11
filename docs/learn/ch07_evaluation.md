# ch07 — 평가 방법론 (가장 중요한 챕터)

> 대응 노트북: `notebooks/ch07_evaluation.ipynb` — **random score로 PA-F1 SOTA 만들기**

## Point Adjustment(PA)가 만든 신기루

PA 프로토콜 (Xu et al. 2018 이후 관행): "이상 이벤트 안의 한 지점이라도 탐지하면
이벤트 **전체**를 탐지한 것으로 간주"하고 F1을 계산한다.

**문제** (Kim et al., AAAI 2022, arXiv:2109.05257): 이벤트가 길수록 한 지점만 우연히
맞혀도 전체가 TP로 바뀐다. 그 결과 —

```python
# 이 저장소에서 직접 재현 (tests/test_metrics.py::test_pa_inflates_random_scores)
scores = rng.random(2000)          # 완전 무작위 점수
pa_f1  = 0.9+                      # "SOTA급"
standard_f1 = 0.05                 # 실제 성능
vus_pr = 0.3                       # 주지표는 속지 않는다
```

수년간 SMAP/MSL/SMD 리더보드의 "F1 0.95+" 보고 다수가 이 프로토콜 위에 서 있었다.
그래서 이 저장소는 **PA-F1을 기본 계산에서 제외**하고 `--legacy-pa` 플래그 + 경고로만
제공한다 (CLAUDE.md §10-4). [지표 괴리 차트](../leaderboard/index.md)가 이를 시각화한다.

## 벤치마크 자체의 결함

Wu & Keogh (TKDE 2021): 기존 벤치마크는 (1) 사소한(trivial) 이상, (2) 잘못된 라벨,
(3) 비현실적 이상 밀도, (4) run-to-failure 편향을 갖는다. UCR Anomaly Archive는
이에 대한 응답이고, TSB-AD (NeurIPS 2024)는 40개 데이터셋을 재라벨링·큐레이션했다.

## VUS — 이 저장소의 주지표

Range-AUC (Paparrizos et al., VLDB 2022): 라벨 경계에 sqrt 경사 버퍼를 두고
existence(이벤트 발견)와 overlap(포인트 겹침)을 함께 반영한 TPR/FPR로 AUC를 계산.
**VUS(Volume Under Surface)**는 버퍼 폭 0..L에 대한 Range-AUC의 평균 —
버퍼 폭 선택에도 강건한 threshold-free 지표다.

- **VUS-PR**: 주지표. 이상이 희소한 TSAD에서 ROC보다 PR이 분별력 있다.
- 구현은 자체 작성 + TSB-AD 공식 구현과 1e-9 수준 일치 검증
  (`tests/test_metrics_agreement.py`).

## 보조 지표들

- **affiliation-F1** (Huet, KDD 2022): 예측-이벤트 간 시간 거리 기반 — 경계 오차에 관대
- **range-F1** (Tatbul, NeurIPS 2018): existence+overlap 가중 구간 정밀도/재현율
- **event-F1**: 이벤트 수준 재현율 × 포인트 수준 정밀도
- **standard-F1**: 가장 엄격한 point 단위 (참고용 best-F1은 oracle 임계값)

지표마다 "무엇을 이상 탐지 성공으로 볼 것인가"의 정의가 다르다.
**단일 지표 순위를 믿지 말고, 지표 간 괴리 자체를 정보로 읽어라.**
