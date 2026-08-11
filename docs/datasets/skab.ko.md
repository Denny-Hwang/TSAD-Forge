# SKAB (Skoltech Anomaly Benchmark)

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
