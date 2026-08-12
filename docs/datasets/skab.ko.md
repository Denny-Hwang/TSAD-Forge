# SKAB (Skoltech Anomaly Benchmark)

**어떤 데이터인가?** Skoltech의 물 순환 물리 테스트베드 — 펌프·밸브와 8개 센서(진동, 압력, 전류, 유량, 온도). 35개 짧은 실험마다 실제 물리 결함(밸브 폐쇄, 캐비테이션 등)을 주입했다. *물리적으로 유도된 정밀 시점* 결함을 가진 드문 벤치마크.

- **출처**: https://github.com/waico/SKAB (Katser & Kozitsin, 2020)
- **구성**: 물 순환 테스트베드 8센서, 실험 35개(valve1/valve2/other) + 정상 실험 1개, 1초 샘플링
- **라이선스**: 저장소는 AGPL-3.0 — **코드는 절대 복사하지 않는다**. 데이터 csv만 사용
  (원 저자들이 데이터 자유 이용을 명시; 인용 필수).
- **다운로드**: `tsad-forge download skab` 또는 `--subset valve1`
- **로더**: `load_skab(experiment="valve1/0")` — train은 `anomaly-free.csv`

## 알려진 결함

- **실험별 짧은 길이**(~1천 스텝) — 딥러닝 모델에는 데이터 부족.
- **changepoint 열 별도**: 본 로더는 anomaly 라벨만 사용, changepoint는 미사용.
- train(anomaly-free)과 test 실험의 운전 조건이 달라 covariate shift 존재.

## 간단 EDA (로컬 데이터 기준)

라벨된 이상 구간을 음영 처리한 test 채널 샘플과 이벤트 길이 분포. 데이터 다운로드 후 `tsad-forge viz`로 재생성됩니다.

<iframe src="../../../assets/eda/skab.html" width="100%" height="720" frameborder="0"></iframe>
