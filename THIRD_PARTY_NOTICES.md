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

(M3 이후 stumpy, statsmodels, PyTorch 등 추가 시 이 표를 갱신한다.)

## 데이터셋

데이터셋은 저장소에 커밋하지 않는다. 출처·라이선스·인용 정보는
`tsad_forge/data/registry.py`와 `docs/datasets/`의 데이터셋 카드에 기록한다.
