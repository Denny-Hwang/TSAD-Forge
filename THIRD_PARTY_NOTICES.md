# Third-Party Notices

이 문서는 TSAD-Forge에 도입된 모든 외부 코드의 라이선스 대장이다.

**절차 (CLAUDE.md §3 필수)**: 코드 도입 전 원저장소의 LICENSE 파일을 실제로 확인 →
아래 표에 기록 → vendored 파일 상단에 원 저작권 고지 유지.

- **vendored (코드 복사)**: MIT / BSD / Apache-2.0 만 허용
- **pip 의존성 / 어댑터**: 라이선스 무관하게 허용하되 기록
- **재구현**: 논문 기반 자체 구현. 원 논문과의 차이를 해당 모듈 docstring에 명시

## 대장

| 이름 | URL | 커밋/버전 | 라이선스 | 도입 방식 | 비고 |
|---|---|---|---|---|---|
| NumPy | https://github.com/numpy/numpy | pip (>=1.26) | BSD-3-Clause | pip 의존성 | |
| pandas | https://github.com/pandas-dev/pandas | pip (>=2.0) | BSD-3-Clause | pip 의존성 | |
| scikit-learn | https://github.com/scikit-learn/scikit-learn | pip (>=1.4) | BSD-3-Clause | pip 의존성 | |
| PyYAML | https://github.com/yaml/pyyaml | pip (>=6.0) | MIT | pip 의존성 | |
| PyArrow | https://github.com/apache/arrow | pip (>=15.0) | Apache-2.0 | pip 의존성 | |
| TSB-AD (evaluation) | https://github.com/TheDatumOrg/TSB-AD | e0975a5f7d3e | Apache-2.0 (LICENSE 원문 확인) | vendored (`tsad_forge/evaluation/_vendor/tsb_ad/`) | 자체 지표 구현의 수치 일치 검증 기준 + affiliation-F1 계산. 원 코드 무수정 |
| SPOT/DSPOT (Siffer et al., KDD 2017) | (참조 구현 GPL-3 — 코드 미사용) | — | — | 논문 수식 기반 자체 구현 (`evaluation/thresholding.py`) | GPL 코드 미참조·미복사 |
| stumpy | https://github.com/TDAmeritrade/stumpy | pip (>=1.12) | BSD-3-Clause | pip 의존성 | Matrix Profile |
| statsmodels | https://github.com/statsmodels/statsmodels | pip (>=0.14) | BSD-3-Clause | pip 의존성 | STL |
| PyTorch | https://github.com/pytorch/pytorch | pip (extras: dl) | BSD-style | pip 의존성 | Gen3-5 |
| tabulate | https://github.com/astanin/python-tabulate | pip (>=0.9) | MIT | pip 의존성 | 리더보드 |

## Gen3–4 모델 라이선스 감사 기록 (M4)

원 저장소 코드는 일절 복사하지 않고 **전부 논문 기반 자체 재구현**했다
(원 논문과의 차이는 각 모듈 docstring에 명시). 감사 결과:

| 모델 | 원 저장소 | 라이선스 확인 결과 | 처리 |
|---|---|---|---|
| AE | (일반 기법) | — | 재구현 |
| LSTM-AD (Malhotra 2015) | 공식 구현 미공개 | — | 재구현 |
| LSTM-P (Telemanom, khundman/telemanom) | 자체 커스텀 라이선스 ("NASA open source") | 비표준 라이선스 | 재구현 (코드 미사용, 라벨 데이터만 사용) |
| Donut VAE (NetManAIOps/donut) | 저장소 라이선스 파일 없음 | 불명 → 복사 금지 | 재구현 |
| DAGMM | 공식 구현 미공개 (비공식 다수) | 불명 → 복사 금지 | 재구현 |
| OmniAnomaly (NetManAIOps/OmniAnomaly) | MIT (단, TF1 의존) | MIT | 재구현 선택 (단순화; docstring 명시) |
| USAD | 공식 구현 비공개(BSD 비공식) | 불명 → 복사 금지 | 재구현 |
| GDN (d-ailin/GDN) | MIT | MIT | 재구현 선택 |
| MTAD-GAT | 공식 구현 미공개 (MS 비공식 MIT) | — | 재구현 |
| Anomaly Transformer (thuml) | MIT | MIT | 재구현 선택 (minimax 단순화) |
| TranAD (imperial-qore/TranAD) | BSD-3-Clause | BSD-3 | 재구현 선택 |
| DCdetector (DAMO-DI-ML) | 라이선스 파일 확인 필요 | 불명 → 복사 금지 | 재구현 |
| TimesNet (thuml/Time-Series-Library) | MIT | MIT | 재구현 선택 (TimesBlock 축소) |
| S-H-ESD (Twitter AnomalyDetection) | R 패키지 GPL-3 — 코드 미사용 | — | GPL-3 | 논문 수식 기반 자체 구현 (`gen1_statistical/sesd.py`) | Hochenbaum et al. 2017 |
| Spectral Residual (Ren et al., KDD 2019) | (MS 저장소 참조 안 함) | — | — | 논문 수식 기반 자체 구현 (`gen2_classical_ml/spectral_residual.py`) | Azure 프로덕션 알고리즘 |
| HBOS | (참조 구현 미사용) | — | — | 논문 기반 자체 구현 | Goldstein & Dengel 2012 |
| rrcf (pip) | https://github.com/kLabUM/rrcf | — | MIT | **도입 보류** — 유지보수 중단(pkg_resources 비호환), 재구현 예정 (roadmap) | |
| MGAB 데이터 | https://github.com/MarkusThill/MGAB | master | CC0 1.0 | 다운로드 스크립트 (데이터 미커밋) | |
| MBA 데이터 | https://github.com/imperial-qore/TranAD (data/MBA) | main | 저장소 BSD-3; PhysioNet ODC-BY | 다운로드 스크립트 (데이터 미커밋) | |
| MERLIN | 공개 참조 구현 라이선스 불명 | 불명 → 복사 금지 | **도입 보류** (Gen2 목록에서 제외) |

(M3 이후 stumpy, statsmodels, PyTorch 등 추가 시 이 표를 갱신한다.)

## 데이터셋

데이터셋은 저장소에 커밋하지 않는다. 출처·라이선스·인용 정보는
`tsad_forge/data/registry.py`와 `docs/datasets/`의 데이터셋 카드에 기록한다.
