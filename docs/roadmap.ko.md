# 연구 로드맵 & 제안

무엇을 조사했고, 무엇을 추가했으며, 다음에 무엇을 추가할지를 후보별 **라이선스 판정**과
함께 정리한다. 원칙은 저장소 규칙 그대로다: vendored 코드/데이터는 permissive만,
제한 데이터는 신청 안내만.

## 이번 라운드에 추가 (검증 후 반영 완료)

| 추가 항목 | 종류 | 이유 | 라이선스 판정 |
|---|---|---|---|
| **MGAB** | 데이터셋 | TSB-AD도 쓰는 카오스 동역학 스트레스 테스트; 육안으로 안 보이는 이상 | **CC0 1.0** — 원 저장소에서 다운로드 |
| **MBA (MIT-BIH 심전도)** | 데이터셋 | TSAD 논문 부록의 고전 준주기 ECG (TranAD 등) | TranAD 저장소 **BSD-3**; PhysioNet **ODC-BY** |
| **S-H-ESD** (`sesd`) | Gen1 모델 | Twitter 프로덕션 탐지기 (robust STL + ESD 통계량) | 논문 기반 자체 구현 (Twitter R 코드는 GPL-3 — 미접촉) |
| **Spectral Residual** (`spectral_residual`) | Gen2 모델 | Microsoft Azure Anomaly Detector의 핵심 알고리즘 (KDD 2019) | 논문 기반 자체 구현 |
| **HBOS** (`hbos`) | Gen2 모델 | 가장 빠른 실무 baseline (PyOD 단골) | 논문 기반 자체 구현 |
| **ForgeEnsemble** (`ensemble_simple`) | 제안 baseline | 저비용·이질 탐지기 4종(sub_pca, sub_knn, iforest, SR)의 순위 합의; outlier 앙상블 이론상 다양성이 단일 멤버 평균을 이긴다 | 자체 기법, Apache-2.0 |

제안은 의도적으로 겸손하다: **ForgeEnsemble은 Gen0(baseline)으로 등록** — 신규성 주장이
아니라 "발표되는 모델이라면 최소한 이겨야 할 기준점"이다. 실제로 몇 세대가 이걸 넘는지
리더보드에서 확인하라.

## 다음 추가 권장 — 데이터셋

| 후보 | 무엇인가 | 라이선스/접근 | 블로커와 계획 |
|---|---|---|---|
| **Exathlon** (VLDB 2021) | 주입·근본원인 라벨이 달린 Spark 클러스터 트레이스; 최고 수준 주석의 *설명가능 AD* 벤치마크 | 저장소 Apache-2.0; 데이터는 외부 다운로드 | 용량 큼(GB대); 다운로더+로더 추가, full 프로파일 전용 |
| **CATS** (Solenix 2023) | 5백만 포인트 우주선 제어계 시뮬레이션, 통제된 이상 200개, 무오염 train | **CC BY 4.0** (Zenodo) | 이 환경에서 Zenodo 접근 불가 — Yahoo처럼 수동 배치 가이드+로더로 선반영 가능 |
| **GHL** (Kaspersky) | 가스오일 가열 루프 ICS 시뮬레이션 + 공격 | 무료지만 등록 필요 | 신청 안내 카드 + 로더만 |
| **Genesis** (HU Berlin) | 픽앤플레이스 설비 PLC 신호 | CC BY-SA 4.0 | 데이터 사용엔 문제없음; 호스트 확인 필요 |
| **NAB 전체** | 현재 58개 중 5개만 수록 | 데이터 자유 | `download nab --subset all`을 clone 안내로 확장 |
| **UCR / TSB-AD 전체** | 이미 통합됨; 이 개발 환경의 egress 정책만이 장애물 | 자유 / Apache-2.0 | 사용자 머신에서 `tsad-forge download ucr / tsb_ad_u` 실행 — 로더·lite 항목은 이미 연결됨 |

## 다음 추가 권장 — 모델

| 후보 | 세대 | 왜 중요한가 | 계획/라이선스 |
|---|---|---|---|
| **DAMP** (Lu, Keogh et al., KDD 2022) | Gen2 | 스트리밍 left-Matrix-Profile; UCR류에서 강력·고속 | 참조 MATLAB 라이선스 불명 → 논문 재구현 |
| **SAND** (Boniol & Paparrizos, VLDB 2021) | Gen2 | 스트리밍 부분수열 클러스터링; TSB-AD 상위권 | 공식 구현 라이선스 확인 후 결정 |
| **RRCF** (Guha et al., ICML 2016) | Gen2 | AWS Kinesis 프로덕션 알고리즘; 진짜 스트리밍 | `rrcf` pip(MIT)은 유지보수 중단으로 최신 setuptools에서 깨짐(`pkg_resources`) → ~150줄 재구현 |
| **SR-CNN** (Ren et al., KDD 2019) | Gen3 | SR saliency 위 학습 임계값; Azure 파이프라인 완성형 | 우리 `spectral_residual`에 소형 conv 헤드 추가 |
| **CARLA** (Darban et al., 2024) | Gen4 | 대조 표현학습 TSAD, 최근 강세 | 논문 재구현 |
| **PatchTST/PatchAD 계열** | Gen4 | 패치 토큰화는 현재 최강 TS 백본 | TimesNet처럼 축소 재구현 |
| **SigLLM / LLM zero-shot** | Gen5 | LLM을 탐지기로 (2024–) — 비싸지만 설정 제로 | `foundation` extra 뒤의 어댑터 |
| **TranAD faithful** (BSD-3) | Gen4 | 현재는 축소 재구현 — BSD-3라 원본 vendoring 가능; MambaTSAD처럼 faithful/compact 쌍으로 단순화 효과 정량화 | vendored 쌍 구성 |

## 다음 추가 권장 — 방법론 (실무 기반)

1. **스트리밍/온라인 트랙**: 실제 배포는 포인트 단위 채점이다. `score_online()`
   프로토콜(고정 메모리, 단일 패스)을 추가하고 그 조건에서 재순위화 —
   RRCF/DAMP/DSPOT이 1급 시민이 되고 배치 transformer는 아니게 된다.
2. **오염 강건성 곡선**: 합성 생성기의 `contamination`(이미 지원)을 스윕해 모델별
   VUS-PR vs train 오염도 곡선 보고 — ch09 산업 질문에 대한 직접 답.
3. **임계값 이전(transfer) 평가**: 지금은 시계열별 임계값이다. 실무는 여러 머신에
   임계값 하나가 필요하다 — *전역* SPOT/conformal 임계값으로 평가하고 성능 하락 보고.
4. **조기 탐지 지연 지표**: NAB처럼 이벤트 시작→첫 경보까지의 평균 지연(스텝)을
   보조 컬럼으로 추가.
5. **비용 정규화 리더보드**: VUS-PR/log-runtime은 이미 시각화됨 — 정렬 가능한
   리더보드 컬럼으로 승격.

## 비권장 (이유와 함께)

- **Yahoo S5 / SWaT / WADI 재배포** — 라이선스 금지; 신청 안내로 대체.
- **NAB 채점 코드, SKAB 코드, Twitter AnomalyDetection 코드** — AGPL/GPL; 데이터만
  사용(NAB/SKAB)하거나 논문 재구현(S-H-ESD).
- **MERLIN / DADA vendoring** — 감사 시점 라이선스 불명; 상류 명확화까지
  THIRD_PARTY_NOTICES에 추적.
