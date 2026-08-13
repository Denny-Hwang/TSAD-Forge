# MGAB (Mackey-Glass Anomaly Benchmark)

**어떤 데이터인가?** 카오스적인 Mackey-Glass 지연 미분방정식에서 생성한 단변량 시계열
10개 — 각각에 **육안으로는 보이지 않는** 이상 10개가 합성 삽입되어 있다. 시각적
outlier 탐지가 아니라 시간 동역학을 실제로 모델링하는지 검증하는 스트레스 테스트.

- **출처**: https://github.com/MarkusThill/MGAB (Thill, Konen & Bäck, 2020)
- **구성**: 10개 시계열 × 10만 포인트; 열: `value`, `is_anomaly`, `is_ignored`
- **라이선스**: **CC0 1.0 (퍼블릭 도메인)** — 완전 재배포 가능
- **다운로드**: `tsad-forge download mgab` (또는 `--subset 1,2`)
- **로더**: `load_mgab(series=1)`

## 알려진 결함 / 주의

- **공식 train/test 분할 없음** — 로더는 가장 긴 무이상 접두 구간(최대 30%)을 train으로
  사용 (문서화된 편차).
- 순수 합성 카오스: *동역학 모델링* 주장 검증에는 탁월하지만, 실데이터의 센서 노이즈·
  드리프트·레짐 변화에 대해서는 아무것도 말해주지 않는다.
- 원 벤치마크가 제외하는 전이 구간 마스크(`is_ignored`)는 meta에 기록만 하고 지표에는
  미적용.

## 간단 EDA (로컬 데이터 기준)

라벨된 이상 구간을 음영 처리한 test 채널 샘플과 이벤트 길이 분포. 데이터 다운로드 후 `tsad-forge viz`로 재생성됩니다.

<iframe src="../../../assets/eda/mgab.html" width="100%" height="720" frameborder="0"></iframe>
