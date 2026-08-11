# ch08 — 임계값과 결정 (Thresholding)

> 대응 노트북: `notebooks/ch08_thresholding.ipynb`

## 왜 임계값이 모델과 분리되어 있나

연속 점수 \(s_t\)를 이진 결정으로 바꾸는 것은 **운영상의 의사결정**이다
(오탐 비용 vs 미탐 비용). 같은 모델이라도 배포 환경에 따라 다른 임계값이 맞다.
그래서 이 저장소는 임계값을 `evaluation/thresholding` 모듈로 분리하고,
threshold-free 지표(VUS)와 임계값 기반 지표(F1 계열)를 분리 보고한다.

## EVT와 SPOT/DSPOT (Siffer et al., KDD 2017)

극단값 이론(Extreme Value Theory)의 Pickands–Balkema–de Haan 정리: 충분히 높은
초기 임계값 \(t\)를 넘는 초과분 \(y = x - t\)는 **분포와 무관하게** 일반화 파레토
분포(GPD)로 수렴한다:

\[ P(y > z) \approx \left(1 + \frac{\gamma z}{\sigma}\right)^{-1/\gamma} \]

GPD를 Grimshaw MLE로 적합하면 목표 초과 확률 \(q\)(예: 10⁻⁴)의 임계값을
**분포 가정 없이** 계산할 수 있다:

\[ z_q = t + \frac{\sigma}{\gamma}\left(\left(\frac{qn}{N_t}\right)^{-\gamma} - 1\right) \]

- **SPOT**: 정적 분포 가정 — `spot_threshold`
- **DSPOT**: 이동평균 drift 제거 후 SPOT — 비정상(non-stationary) 점수 스트림용

원 저자 참조 구현은 GPL-3이므로 **논문 수식만으로 자체 구현**했다 (라이선스 대장 참조).

## Split-Conformal

보정(calibration) 점수 \(s_1..s_n\)(정상 가정)의
\(\lceil (n+1)(1-\alpha) \rceil\)-번째 순서통계량을 임계값으로 쓰면, 교환가능성 하에서
**오탐률 ≤ α의 유한표본 보장**을 얻는다. 분포 가정도, 점근 근사도 없다.
이 저장소에서는 train 점수를 보정 표본으로 사용한다 (`conformal_threshold`).

## 실무 가이드

| 상황 | 권장 |
|---|---|
| 점수 분포가 안정적, 보수적 운영 | conformal (α = 목표 오탐률) |
| 극단 꼬리 이상, 이론적 근거 필요 | SPOT (q = 초과 확률) |
| 점수에 drift 존재 | DSPOT |
| 빠른 프로토타이핑 | quantile (q=0.99) |

**best-F1(oracle 임계값)은 리더보드 참고용일 뿐 배포 시 재현 불가능하다** —
test 라벨을 몰래 본 값이기 때문이다.
