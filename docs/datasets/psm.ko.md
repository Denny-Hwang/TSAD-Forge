# PSM (Pooled Server Metrics, eBay)

**어떤 데이터인가?** eBay 프로덕션 인프라의 Pooled Server Metrics — 다수 서버 노드의 CPU/메모리/네트워크 25채널 집계, 분 단위, train 13주 + 라벨된 test 8주. *대규모 IT 운영* 벤치마크.

- **출처**: https://github.com/eBay/RANSynCoders (Abdulaal et al., KDD 2021)
- **구성**: 25차원 서버 지표, train 13만+ / test 8.7만+ 스텝, 1분 샘플링
- **라이선스**: 저장소 Apache-2.0
- **다운로드**: `tsad-forge download psm`
- **로더**: `load_psm()`

## 알려진 결함

- **train 결측**: train.csv에 NaN 다수 — 로더가 선형 보간(문서화된 기본 동작).
- **거친 라벨 경계**: 분 단위 집계 라벨이라 이벤트 경계가 부정확할 수 있음.
- 선행 보고 다수가 PA-F1 기준 — 비교 불가.

## 간단 EDA (로컬 데이터 기준)

라벨된 이상 구간을 음영 처리한 test 채널 샘플과 이벤트 길이 분포. 데이터 다운로드 후 `tsad-forge viz`로 재생성됩니다.

<iframe src="../../../assets/eda/psm.html" width="100%" height="720" frameborder="0"></iframe>
