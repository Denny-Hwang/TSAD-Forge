# ch09 — 산업 적용 가이드: 반도체 FDC/ATE 관점

> 대응 노트북: `notebooks/ch09_industrial.ipynb` — BYOD 워크플로

## 벤치마크와 현장의 간극

| 벤치마크 가정 | 반도체 FDC/ATE 현실 |
|---|---|
| 정상 train 존재 | 레시피·장비 PM 주기마다 "정상"이 바뀜 |
| 균일 샘플링 | 스텝별 샘플링 레이트 상이, 이벤트 구동 로깅 |
| 시계열 하나가 길게 연속 | 웨이퍼/랏 단위로 조각난 짧은 시계열 다수 |
| point 라벨 | "이 랏이 불량" 수준의 약한 라벨 (weak label) |

## Regime vs Fault — 가장 흔한 실패 원인

레시피 전환, PM 직후, 계절 변화 같은 **운전 영역(regime) 변화**는 통계적으로는
이상처럼 보이지만 fault가 아니다. 대응 전략:

1. regime 메타데이터(레시피 ID 등)로 데이터를 분리해 regime별 모델 학습
2. DSPOT처럼 drift 적응형 임계값 사용 (ch08)
3. 변화점 탐지(changepoint)와 이상탐지를 분리 — SKAB 데이터셋이 두 라벨을 모두 제공

## 라벨 희소·불규칙 샘플링·cold start

- **라벨 희소**: threshold-free 지표(VUS)로 모델을 고르고, conformal로 오탐률을 보장
- **불규칙 샘플링**: SSM의 Δ 이산화 가정(ch06)이 깨진다 — 리샘플링하거나
  간격을 특징으로 추가. Matrix Profile도 균일 샘플링 가정임에 주의.
- **cold start**: 신규 장비·신규 레시피에는 train이 없다 — 파운데이션 모델
  zero-shot(ch06) 또는 유사 장비 전이가 후보. SECOM 카드(docs/datasets/secom.md)는
  "시계열이 아닌 FDC 요약 통계" 문제와의 구분 예시다.

## BYOD 워크플로

```bash
# 라벨 없는 CSV: 스코어 + 임계값 결정만 출력
tsad-forge run --model sub_pca --data my_sensor.csv

# label 열이 있으면 전체 지표 산출
tsad-forge run --model iforest --data my_labeled.csv --seed 0

# EVT 임계값으로
tsad-forge run --model sub_pca --data my_sensor.csv --config configs/default.yaml
```

CSV 규약: 숫자 열 = 채널, `timestamp` 열(옵션) = 정렬, `label` 열(옵션) = 0/1.
결측은 선형 보간된다. 첫 이상 이전 구간이 train으로 쓰인다 (unsupervised 가정 유지).

**시작 조합 권장**: `sub_pca` + `conformal` — 단순·빠름·오탐 보장. 그 다음에야
딥러닝을 검토하라 (리더보드가 그 근거다).
