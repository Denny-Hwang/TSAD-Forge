# CLAUDE.md — TSAD-Forge 프로젝트 사양서

## 0. 프로젝트 정체성

- **이름**: TSAD-Forge
- **부제**: A Reproducible Benchmarking and Learning Ecosystem for Time-Series Anomaly Detection
- **라이선스**: Apache-2.0 (저장소 자체 코드)
- **목적**: 비지도 시계열 이상탐지(TSAD)의 기술 발전을 세대(Generation)별로 정리하고,
  (1) 공식 데이터셋에 대해 세대별 대표 기법을 동일 프로토콜로 재현·평가하고,
  (2) 사용자가 git clone 후 공식 데이터 재현 또는 자기 데이터(BYOD) 적용을 즉시 할 수 있게 하며,
  (3) 신규 학습자를 위한 이론 학습 자료를 제공하고,
  (4) 전체 벤치마크 결과를 GitHub Pages에 인터랙티브 시각화로 게시한다.
- **핵심 설계 철학**:
  - 평가 방법론 우선: point adjustment(PA)의 성능 부풀림(Kim et al., AAAI 2022, arXiv:2109.05257)과
    벤치마크 결함(Wu & Keogh, TKDE 2021, arXiv:2009.13807)을 전제로, VUS-PR(Paparrizos et al.,
    VLDB 2022)을 주지표로 삼는다. TSB-AD(Liu & Paparrizos, NeurIPS 2024)의 프로토콜을 참조 표준으로 한다.
  - 단순 baseline 존중: Sub-PCA, Matrix Profile, IForest 같은 단순 기법이 딥러닝을 이기는 경우가
    많다는 사실(TSB-AD, Sarfraz et al. ICML 2024)을 리더보드에 정직하게 반영한다.
  - 라이선스 청정성: Apache-2.0 배포가 가능하도록 permissive 라이선스 코드만 vendored한다.

## 1. 저장소 구조

```
TSAD-Forge/
├── CLAUDE.md                     # 본 사양서
├── README.md                     # 프로젝트 소개 (영문, 한국어 요약 링크)
├── LICENSE                       # Apache-2.0
├── THIRD_PARTY_NOTICES.md        # vendored/의존 코드 라이선스 대장
├── CITATION.cff
├── pyproject.toml                # 패키지: tsad_forge, extras: [dl, foundation, docs, dev]
├── environment.yml               # conda 환경 (python 3.11, pytorch 2.6 cu124)
├── Dockerfile
├── tsad_forge/                   # 핵심 패키지
│   ├── data/                     # 데이터셋 레지스트리·로더·다운로드
│   │   ├── registry.py           # 데이터셋 메타(출처, 라이선스, 체크섬, 인용)
│   │   ├── loaders/              # 데이터셋별 로더 (통일 스키마로 반환)
│   │   └── download.py           # CLI: tsad-forge download <dataset>
│   ├── models/                   # 세대별 모델 (통일 API)
│   │   ├── base.py               # BaseDetector: fit(X), score(X)->anomaly_score
│   │   ├── gen1_statistical/     # CUSUM, EWMA, PCA-T2SPE, Sub-PCA, STL-residual, POLY
│   │   ├── gen2_classical_ml/    # LOF, OCSVM, IForest, KNN, MatrixProfile(stumpy), (MERLIN)
│   │   ├── gen3_dl_recon/        # AE, LSTM-AD, Telemanom형 LSTM-P, VAE(Donut형), DAGMM,
│   │   │                         # OmniAnomaly, USAD
│   │   ├── gen4_graph_transformer/ # GDN, MTAD-GAT, AnomalyTransformer, TranAD, DCdetector,
│   │   │                         # TimesNet(-AD)
│   │   └── gen5_ssm_foundation/  # MambaTSAD(Chen et al. 2024 재현 + bugfix 변형),
│   │                             # MOMENT(zero-shot/FT), Chronos/TimesFM 예측잔차 어댑터,
│   │                             # (DADA — 라이선스 확인 후)
│   ├── evaluation/
│   │   ├── metrics.py            # VUS-PR/VUS-ROC(주지표), AUC-PR/ROC, affiliation-F,
│   │   │                         # range-F1, event-F1, standard-F1; PA-F1은 legacy 플래그 전용
│   │   ├── thresholding.py       # SPOT/DSPOT(EVT), conformal, quantile, best-F1(참고용)
│   │   └── protocol.py           # train/test 분할, 시드, 스코어 저장 규약
│   ├── synthetic/                # 합성 이상 주입기 (spike/level shift/pattern/frequency/
│   │                             # contextual + contamination 실험용)
│   ├── runner/                   # 실험 실행기 (config 기반, hydra 또는 yaml+argparse)
│   └── viz/                      # 시각화 생성 (아래 §6)
├── configs/                      # 실험 config (모델×데이터셋 매트릭스)
├── benchmarks/
│   ├── results/                  # parquet + JSON (모델, 데이터셋, 지표, 시드, 실행시간, 커밋해시)
│   └── run_all.py                # 전체/부분 벤치마크 실행 스크립트
├── docs/                         # 학습 자료 + GitHub Pages 소스 (MkDocs Material)
│   ├── learn/                    # 이론 학습 트랙 (§7)
│   ├── leaderboard/              # 자동 생성 리더보드 페이지
│   └── datasets/                 # 데이터셋 카드
├── notebooks/                    # 튜토리얼 노트북 (§7)
├── scripts/
├── tests/                        # pytest
└── .github/workflows/            # CI(스모크 테스트) + Pages 빌드·배포
```

## 2. 데이터셋 레이어

**통일 스키마**: 모든 로더는 `TSADDataset(train: np.ndarray[T,D], test: np.ndarray[T,D],
labels: np.ndarray[T], meta: dict)` 를 반환. 단변량은 D=1. meta에는 name, source_url,
license, citation, sampling 정보, 이상 이벤트 수/길이 통계 포함.

**포함 데이터셋(우선순위 순)과 취급 원칙**:

| 데이터셋 | 접근 | 취급 |
|---|---|---|
| NASA SMAP/MSL | HuggingFace 미러 `appleparan/telemanom` | 다운로드 스크립트 |
| SMD (Server Machine Dataset) | OmniAnomaly/TranAD GitHub 미러 | 다운로드 스크립트 |
| UCR Anomaly Archive (250개) | Keogh 공식 배포 | 다운로드 스크립트 + 안내 |
| TSB-AD-U / TSB-AD-M | TheDatumOrg/TSB-AD | 다운로드 스크립트, 프로토콜 호환 로더 |
| NAB (Numenta) | GitHub (AGPL 데이터 아님, 데이터는 자유) | 다운로드 스크립트 |
| PSM (eBay) | GitHub 공개 | 다운로드 스크립트 |
| SKAB | GitHub 공개 | 다운로드 스크립트 |
| Yahoo S5 | 신청 필요, 재배포 금지 | 신청 안내 문서 + 로컬 배치 시 로더만 제공 |
| SWaT / WADI | iTrust 신청 필요, 재배포 금지 | 동일 (안내 + 로더만) |
| SECOM (UCI) | UCI 공개 | 반도체 참고용(시계열 아님, 문서에 성격 명시) |

**주의사항 문서화**: 각 데이터셋 카드에 알려진 결함(SMAP/MSL의 준이진 채널·희소 라벨,
SMD의 이벤트 길이 분산, PA 기반 선행 보고와의 비교 불가성 등)을 명시한다.

**BYOD(Bring Your Own Data)**: `tsad-forge run --data path/to/your.csv --model iforest`
형태로 CSV/parquet(단변량·다변량, timestamp 열 옵션)를 즉시 평가·시각화. 라벨이 없으면
스코어+임계값 결과만 출력하고, 라벨 열이 있으면 전체 지표 산출.

## 3. 모델 레이어 — 세대(Generation) 분류

모든 모델은 `BaseDetector` 상속: `fit(train)`, `score(test) -> np.ndarray[T]` (연속 이상 점수).
임계값 적용은 모델이 아니라 `thresholding` 모듈의 책임 (관심사 분리).

- **Gen 1 — Statistical (1930s–2000s)**: CUSUM, EWMA, Hotelling T², PCA-T²/SPE, Sub-PCA,
  STL 잔차, POLY. 자체 구현(참조: statsmodels).
- **Gen 2 — Classical ML (2000–2016)**: LOF, OC-SVM, IForest, KNN, Sub-KNN
  (scikit-learn, BSD), Matrix Profile/discord (stumpy, BSD-3), MERLIN(구현체 라이선스
  확인 후 자체 구현 여부 결정).
- **Gen 3 — DL 재구성/예측 (2015–2020)**: AE, LSTM-AD, LSTM 예측+동적 임계값(Telemanom형),
  VAE(Donut형), DAGMM, OmniAnomaly, USAD. 원저장소 라이선스 확인 후 permissive면 어댑터
  vendored, 아니면 논문 기반 자체 재구현(재구현 시 원 논문과의 차이를 docstring에 명시).
- **Gen 4 — Graph/Transformer (2020–2023)**: GDN, MTAD-GAT, Anomaly Transformer, TranAD
  (BSD-3 확인), DCdetector, TimesNet. 동일 원칙.
- **Gen 5 — SSM/Foundation (2023–)**:
  - MambaTSAD: Chen et al. (IEEE SPL 2024, arXiv:2405.19823) 재현. **두 가지 변형 제공**:
    (a) `faithful` — 원 구현 충실 재현, (b) `fixed` — 알려진 구현 이슈(hidden state 인덱싱,
    CPU/GPU 분기, HP filter 목적함수, AMA 전역 FFT) 수정판. 두 변형을 벤치마크에서 나란히
    비교해 수정 효과를 정량화한다.
  - MOMENT (ICML 2024) zero-shot/fine-tuned 어댑터, Chronos·TimesFM 예측 잔차 어댑터.
  - DADA (ICLR 2025): 공개 저장소 라이선스 확인 후 결정.
  - mamba-ssm 미설치 환경 대비: 순수 PyTorch fallback 구현(느리지만 동작) 제공.

**라이선스 절차(필수)**: 코드 도입 전 원저장소 LICENSE를 fetch하여 확인 →
THIRD_PARTY_NOTICES.md에 [이름, URL, 커밋해시, 라이선스, 도입 방식(vendored/pip/재구현)]
기록 → vendored 파일 상단에 원 저작권 고지 유지.

## 4. 평가 레이어

- **주지표**: VUS-PR. 보조: VUS-ROC, AUC-PR, AUC-ROC, affiliation-F1, range-F1, event-F1,
  standard-F1. **PA-F1은 `--legacy-pa` 플래그로만 계산**되며 출력에 "PA는 성능을 부풀린다
  (random score도 SOTA화됨)" 경고를 붙인다.
- 지표 구현은 자체 구현 + TSB-AD 공식 구현과의 수치 일치 테스트(허용오차 내)로 검증한다.
- **임계값 모듈**: SPOT/DSPOT(EVT, Siffer KDD 2017), split-conformal, quantile.
  임계값별 F1과 threshold-free 지표를 분리 보고.
- **프로토콜**: 시드 3개 이상(딥러닝), 데이터셋별 공식 train/test 분할 준수, z-score 정규화
  기본, 스코어 원본을 results에 저장(지표 재계산 가능하게).

## 5. 벤치마크 실행 체계

- `configs/`에 (모델 × 데이터셋) 매트릭스. 두 프로파일:
  - **lite**: CI/노트북용 소규모 부분집합(UCR 일부 + SMD 3개 머신 + SMAP 3채널), CPU 가능
  - **full**: 전체 매트릭스, 8GB GPU 기준 순차 실행
- 결과 스키마(parquet): model, generation, dataset, channel, seed, metric, value,
  runtime_s(=fit+score), runtime_fit_s, runtime_score_s, peak_mem_mb(호스트 RSS 피크
  증가분; CUDA 실행 시 peak_vram_mb를 JSON 요약에 병기), commit_hash, config_hash,
  timestamp. 집계 시 동일 (model, dataset, channel, seed)의 구 config_hash 결과는
  최신 timestamp로 대체한다(이중 집계 방지).
- 실행 재개(resume) 지원: 이미 결과가 있는 (model, dataset, seed) 조합은 건너뜀.

## 6. 시각화 + GitHub Pages

MkDocs Material + GitHub Actions로 `docs/` → Pages 자동 배포. 시각화는 Plotly(인터랙티브,
HTML 임베드)로 `tsad_forge/viz/`에서 results parquet로부터 자동 생성:

1. **세대별 성능 진화 차트**: x축 세대(Gen1→5), y축 VUS-PR 분포(box/violin) — "세대가
   올라간다고 성능이 오르는가?"를 정직하게 보여주는 핵심 차트
2. **리더보드 테이블**: 데이터셋군별 정렬 가능 테이블(지표 선택 드롭다운)
3. **모델×데이터셋 히트맵**: VUS-PR
4. **Critical Difference 다이어그램**: 평균 순위 기반 (autorank 또는 자체 구현)
5. **성능 vs 비용 산점도**: VUS-PR vs 실행시간/VRAM (실무 배포 관점)
6. **지표 간 괴리 차트**: PA-F1 vs VUS-PR 산점도 — 평가 부풀림을 시각적으로 증명
7. **사례 뷰어**: 대표 시계열 위에 각 세대 모델의 스코어 오버레이 + 정답 구간 음영
8. **데이터셋 카드 차트**: 이상 길이 분포, 채널 통계

## 7. 학습 트랙 (신규 학습자용, docs/learn/)

한국어 본문 + 영어 용어 병기. 챕터당 이론 문서 + 대응 노트북 1개(notebooks/):

- ch01 TSAD 문제 정의: 이상 유형(point/contextual/collective), unsupervised vs
  normality-based, contamination
- ch02 Gen1 통계: 관리도, T²/SPE 유도, STL — 노트북: PCA-T²/SPE를 SMD에 적용
- ch03 Gen2 고전 ML: IForest 원리, Matrix Profile/discord — 노트북: stumpy로 UCR discord
- ch04 Gen3 DL: 재구성 vs 예측, VAE, over-generalization 문제
- ch05 Gen4: 그래프(GDN)와 어텐션(association discrepancy)
- ch06 Gen5: SSM/Mamba(선택적 게이팅, Δ 이산화와 균일 샘플링 가정), 파운데이션 모델과
  zero-shot
- ch07 평가 방법론(가장 중요): PA 부풀림 재현 실험 노트북(random score로 PA-F1 SOTA 만들기),
  VUS/affiliation 소개
- ch08 임계값과 결정: EVT(SPOT/DSPOT), conformal
- ch09 산업 적용 가이드: 반도체 FDC/ATE 관점 — regime vs fault, 라벨 희소, 불규칙 샘플링,
  cold start, BYOD 워크플로
- ch10 벤치마크 읽는 법: 이 저장소 리더보드 해석 가이드 + 참고문헌 전체 목록

## 8. 마일스톤 (Definition of Done 포함)

- **M0 스캐폴딩**: 구조 생성, pyproject/conda/Docker, CI 뼈대, pre-commit(ruff, black),
  BaseDetector + 더미 모델 + 더미 데이터로 end-to-end 파이프라인 1회 통과. DoD: `pytest`
  녹색, `tsad-forge run --model dummy --data synthetic` 동작.
- **M1 데이터 레이어**: 레지스트리 + SMAP/MSL, SMD, UCR, TSB-AD 로더 + 체크섬 + 데이터셋
  카드 초안. DoD: 4개 데이터셋 다운로드→로드→스키마 검증 테스트 통과, 재배포 금지
  데이터셋 안내 문서 존재.
- **M2 평가 레이어**: 전 지표 + 임계값 모듈 + TSB-AD 구현과의 일치 테스트. DoD: 합성
  케이스에 대한 지표 단위 테스트, PA 경고 동작 확인.
- **M3 Gen1–2 모델 + lite 벤치마크**: 통계·고전 ML 전부 + lite 프로파일 실행 + 결과
  parquet 생성. DoD: lite 리더보드 산출.
- **M4 Gen3–4 모델**: 라이선스 감사 후 도입/재구현, 8GB VRAM 검증. DoD: 각 모델 smoke
  테스트 + lite 벤치마크 갱신.
- **M5 Gen5 모델**: MambaTSAD faithful/fixed 두 변형, MOMENT/Chronos 어댑터. DoD: faithful
  vs fixed 비교 결과 생성.
- **M6 full 벤치마크 + 시각화**: full 실행, viz 8종 생성. DoD: results parquet 완결,
  모든 차트 HTML 생성.
- **M7 문서 + Pages**: 학습 트랙 ch01–10, 리더보드 페이지, Actions 배포. DoD: Pages
  로컬 빌드 성공(`mkdocs build`), README에 quickstart 3줄(clone→install→run) 검증.

## 9. 품질·재현성 원칙

- 모든 무작위성 시드 고정, config에 기록. 결과 재현 커맨드를 리더보드 각 행에서 확인
  가능하게(config_hash → configs 링크).
- CI는 lite 프로파일 스모크만(무거운 학습 금지). full 벤치마크는 로컬 수동 실행.
- 문서·코드 주석에서 성능 주장 시 반드시 지표·데이터셋·시드 수를 병기. "SOTA" 단어는
  리더보드 근거 없이 사용 금지.
- 데이터/모델의 알려진 한계(SMAP 준이진 채널, PA 기반 선행 보고와의 비교 불가 등)를
  숨기지 않고 문서화한다.

## 10. 작업 규칙 (Claude Code 운용)

1. 마일스톤 단위로 작업하고, 각 마일스톤 완료 시 Definition of Done 체크리스트를 검증한 뒤
   보고하고 승인을 받는다. 승인 전에 다음 마일스톤으로 넘어가지 않는다.
2. 외부 코드를 가져오기 전에 반드시 해당 저장소의 LICENSE 파일을 실제로 확인한다.
   MIT/BSD/Apache-2.0만 vendored(코드 복사) 허용. GPL/AGPL/라이선스 불명은 코드 복사 금지 —
   pip 의존성 또는 어댑터로만 연결하고 THIRD_PARTY_NOTICES.md에 기록한다.
3. 데이터셋은 절대 저장소에 커밋하지 않는다. 다운로드 스크립트 + SHA256 체크섬 +
   출처/라이선스 문서화만 커밋한다.
4. 평가에서 point adjustment는 기본값으로 절대 사용하지 않는다. VUS-PR이 주지표다.
   PA-F1은 `--legacy-pa` 플래그로만, 경고 문구와 함께 출력한다.
5. 모든 실험은 시드 고정 + config 파일 기반으로 재현 가능해야 하며, 결과는
   benchmarks/results/ 아래 parquet+JSON으로 저장한다.
6. 개발 환경: Python 3.11, PyTorch 2.6 (CUDA 12.4), 단일 8GB GPU(RTX A2000 Laptop) 기준.
   모든 기본 config는 8GB VRAM에서 실행 가능해야 한다.
7. 각 마일스톤마다 pytest 테스트를 함께 작성하고, 커밋 전 테스트를 통과시킨다.
