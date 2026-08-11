# ch03 — Gen2 고전 기계학습 (2000–2016)

> 대응 노트북: `notebooks/ch03_gen2_classical_ml.ipynb` — stumpy로 UCR discord 발견

## 밀도·경계 기반: LOF, OC-SVM, KNN

시계열을 **부분수열 윈도우 벡터**로 임베딩하면 (`embed_windows`) 일반 outlier detection
기법을 그대로 쓸 수 있다:

- **KNN / Sub-KNN**: train 윈도우들과의 k번째 최근접 거리. 단순하지만 단변량에서 매우 강력.
- **LOF**: 국소 밀도 대비 상대 밀도 — 밀도가 불균일한 데이터에서 KNN보다 낫다.
- **OC-SVM**: 정상 데이터를 감싸는 최소 경계 학습. \(\nu\)가 경계 밖 허용 비율을 제어.

## Isolation Forest — "고립되기 쉬운 점이 이상이다"

랜덤 축·랜덤 분할의 트리에서 **이상점은 얕은 깊이에서 고립**된다. 밀도 추정 없이
O(n log n)으로 동작하고 고차원에 강해 실무 기본값으로 가장 널리 쓰인다.
확률적 알고리즘이므로 이 저장소에서는 **시드 3개**로 평가한다 (`stochastic: true`).

## Matrix Profile — discord의 발견

부분수열 간 z-정규화 유클리드 거리의 최소값 프로파일 \(MP_i = \min_{j} d(w_i, w_j)\).
**discord**(가장 가까운 이웃조차 먼 부분수열)가 collective 이상의 정의와 정확히 일치한다.

- 파라미터가 윈도우 길이 하나뿐 — "parameter-free에 가까운" 강건함
- train이 필요 없다 (test self-join) — 진짜 unsupervised
- 반복 이상(같은 이상이 2회 이상)은 서로를 이웃으로 삼아 놓칠 수 있음 → twin freak 문제

UCR Anomaly Archive는 사실상 discord 발견 문제로 설계되어 있어 MP가 특히 강하다.
구현은 stumpy(BSD-3)를 pip 의존성으로 사용한다 (`matrix_profile`).

## MERLIN은 왜 없나

MERLIN(모든 길이의 discord 탐색)은 공개 참조 구현의 라이선스가 불명확해
도입을 보류했다 (CLAUDE.md §10-2 — 라이선스 불명은 코드 복사 금지).
THIRD_PARTY_NOTICES.md에 기록되어 있다.
