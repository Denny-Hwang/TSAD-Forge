# ch01 — TSAD 문제 정의

> 대응 노트북: [`notebooks/ch01_problem.ipynb`](https://github.com/Denny-Hwang/TSAD-Forge/tree/main/notebooks)

## 시계열 이상탐지(Time-Series Anomaly Detection)란

주어진 시계열 \(x_1, \dots, x_T\) (단변량 D=1 또는 다변량 D>1)에서 "정상 거동에서
벗어난 구간"을 찾는 문제다. TSAD-Forge의 모든 모델은 **연속 이상 점수(anomaly score)**
\(s_t \in \mathbb{R}\)를 출력하며, 이진 결정은 별도의 임계값 모듈이 담당한다 (ch08).

## 이상의 유형 (anomaly taxonomy)

| 유형 | 정의 | 예시 |
|---|---|---|
| **Point** | 개별 시점 값이 비정상 | 센서 스파이크 |
| **Contextual** | 값 자체는 정상 범위지만 맥락(시각·주기 위상)상 비정상 | 새벽의 낮 수준 트래픽 |
| **Collective** | 개별 값은 정상이나 부분수열 전체의 패턴이 비정상 | 파형 왜곡, 주기 붕괴 |

`tsad_forge/synthetic/injectors.py`가 이 유형들(spike / level_shift / pattern /
frequency / contextual)을 합성 주입한다 — 노트북에서 직접 만들어 본다.

## Unsupervised vs normality-based

문헌에서 "unsupervised TSAD"는 대부분 **normality-based**(semi-supervised) 설정이다:
train 구간은 "대체로 정상"이라고 가정하고 정상 거동을 학습한 뒤, test에서 벗어남을
점수화한다. 진짜 unsupervised(라벨·정상 가정 모두 없음)는 Matrix Profile discord처럼
test 자체의 self-join으로 동작한다.

## Contamination (오염)

train에 이상이 섞여 있으면(예: NAB의 probationary 구간) 정상 모델이 이상까지 학습해
탐지력이 떨어진다. `generate_synthetic(contamination=...)`으로 이 효과를 실험할 수 있다.
산업 현장(ch09)에서는 "완전히 깨끗한 train"이 거의 존재하지 않으므로 중요한 축이다.

## TSAD가 어려운 진짜 이유

1. **라벨의 희소성과 모호성** — 이상 경계는 annotator마다 다르다 (Wu & Keogh, TKDE 2021).
2. **평가 방법론의 함정** — point adjustment는 random score도 SOTA로 만든다 (ch07).
3. **단순 baseline의 강력함** — Sub-PCA·IForest가 딥러닝을 자주 이긴다 (TSB-AD, NeurIPS 2024).

이 세 가지가 이 저장소의 설계 철학(CLAUDE.md §0)이다.
