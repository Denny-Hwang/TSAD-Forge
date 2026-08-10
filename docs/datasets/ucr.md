# UCR Anomaly Archive (2021)

- **출처**: Wu & Keogh (TKDE 2021) — https://www.cs.ucr.edu/~eamonn/time_series_data_2018/
- **구성**: 250개 단변량 시계열. 파일명에 train 경계와 이상 구간이 인코딩:
  `NNN_UCR_Anomaly_<name>_<trainEnd>_<anomStart>_<anomEnd>.txt`
- **라이선스**: 연구 목적 자유 이용
- **다운로드**: `tsad-forge download ucr` (zip 약 250MB)
- **로더**: `load_ucr(series=1)` 또는 `load_ucr(series="001_UCR_Anomaly_...txt")`

## 설계 의도와 주의

- 이 아카이브는 기존 벤치마크의 결함(사소한 이상, 잘못된 라벨, 과도한 사전 정보)을
  비판하며 만들어졌다 — **시계열당 이상 정확히 1개** 규약.
- 단일 이상 규약 때문에 event-F1 등 event-level 지표는 이 데이터셋에서 이분법적으로
  동작한다. VUS-PR 해석 시 참고.
- DISTORTED/NOISE 접두 파일들은 동일 원본의 변형판 — 완전히 독립적인 250개가 아니다.
