# ch04 — Gen3 딥러닝: 재구성 vs 예측 (2015–2020)

> 대응 노트북: `notebooks/ch04_gen3_dl.ipynb`

## 두 가지 패러다임

| | 재구성 (reconstruction) | 예측 (forecasting) |
|---|---|---|
| 학습 | 윈도우를 압축→복원 | 과거로 다음 값 예측 |
| 점수 | 재구성 오차 | 예측 오차 |
| 대표 | AE, VAE(Donut), USAD | LSTM-AD, LSTM-P(Telemanom) |
| 강점 | collective 이상 | point·급변 이상 |
| 약점 | over-generalization | 느린 드리프트에 둔감 |

## VAE와 재구성 "확률"

Donut (Xu et al., WWW 2018)은 재구성 값이 아니라 **재구성 확률**
\(-\log p(x|z)\)을 점수로 쓴다. 분산까지 학습하므로 "원래 변동이 큰 구간"과
"진짜 이상"을 구분할 수 있다 — 결정적 AE 대비 핵심 장점.

## Over-generalization 문제

용량이 큰 AE는 **이상까지 잘 복원해 버린다**. 대응책이 Gen3 후반 연구의 흐름:

- **DAGMM**: 잠재 공간에 GMM을 두어 "정상 밀도가 낮은 지점"을 에너지로 점수화
- **USAD**: 두 디코더의 적대적 학습으로 재구성 경계를 날카롭게
- **OmniAnomaly**: 확률적 RNN으로 시간 의존적 잠재 분포를 모델링

## 이 저장소의 구현 노트

Gen3 전 모델은 논문 기반 재구현이며 (라이선스 감사: THIRD_PARTY_NOTICES.md),
단순화 사항은 각 모듈 docstring에 명시했다. 예: OmniAnomaly는 normalizing flow와
linear Gaussian SSM prior를 생략한 GRU-VAE다. Telemanom의 동적 임계값은 모델이 아닌
`thresholding` 모듈(SPOT 등)이 담당한다 — 관심사 분리 (CLAUDE.md §3).

**주의**: 이 세대의 원 논문 수치는 대부분 PA-F1 기준으로 보고되었다.
ch07을 읽기 전까지 그 수치를 믿지 마라.
