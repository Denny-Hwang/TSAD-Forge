# NAB (Numenta Anomaly Benchmark)

**어떤 데이터인가?** Numenta가 수집한 짧은 단변량 스트림 58개: AWS 서버 지표, 기계 온도, 도시 교통량, 광고 클릭, 트윗. 라벨은 알려진 사건 주변의 넓은 윈도우다. 조기 탐지에 보상을 주는 *스트리밍* 탐지용 설계.

- **출처**: https://github.com/numenta/NAB (Lavin & Ahmad, ICMLA 2015)
- **구성**: 58개 단변량 시계열(AWS 지표, 온도 센서, 광고 클릭 등) + 이상 윈도우 라벨
- **라이선스**: **코드는 AGPL-3.0 — 절대 복사하지 않는다** (CLAUDE.md §10-2).
  데이터 파일은 자유 이용. 본 저장소는 데이터 csv와 라벨 json만 받는다.
- **다운로드**: `tsad-forge download nab` (lite 부분집합 5개; 전체는 NAB 저장소 clone 권장)
- **로더**: `load_nab(rel_path="realAWSCloudwatch/ec2_cpu_utilization_24ae8d.csv")`

## 알려진 결함

- **train 분할 부재**: NAB는 스트리밍 평가 설계라 공식 train/test 분할이 없다.
  본 저장소는 NAB probationary 규약(앞 15%)을 train으로 쓰며, **train에 이상이 포함될 수
  있다** (contamination) — unsupervised 가정 위반 가능성을 meta에 기록.
- **윈도우 라벨**: 라벨이 점이 아닌 넓은 윈도우 — point-level 지표가 관대해짐.
- Wu & Keogh (TKDE 2021)는 NAB 라벨 품질 자체를 비판 — 결과 해석 주의.

## 간단 EDA (로컬 데이터 기준)

라벨된 이상 구간을 음영 처리한 test 채널 샘플과 이벤트 길이 분포. 데이터 다운로드 후 `tsad-forge viz`로 재생성됩니다.

<iframe src="../../../assets/eda/nab.html" width="100%" height="720" frameborder="0"></iframe>
