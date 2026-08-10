# PSM (Pooled Server Metrics, eBay)

- **출처**: https://github.com/eBay/RANSynCoders (Abdulaal et al., KDD 2021)
- **구성**: 25차원 서버 지표, train 13만+ / test 8.7만+ 스텝, 1분 샘플링
- **라이선스**: 저장소 Apache-2.0
- **다운로드**: `tsad-forge download psm`
- **로더**: `load_psm()`

## 알려진 결함

- **train 결측**: train.csv에 NaN 다수 — 로더가 선형 보간(문서화된 기본 동작).
- **거친 라벨 경계**: 분 단위 집계 라벨이라 이벤트 경계가 부정확할 수 있음.
- 선행 보고 다수가 PA-F1 기준 — 비교 불가.
