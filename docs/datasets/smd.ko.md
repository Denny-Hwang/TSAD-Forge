# SMD (Server Machine Dataset)

**어떤 데이터인가?** 대형 인터넷 기업 서버 5주치 텔레메트리 — 머신당 38채널(CPU 부하, 네트워크, 메모리, 디스크 I/O 등)을 1분 간격 수집. 이상은 도메인 전문가가 주석한 실제 운영 장애다. *다변량 서버 모니터링*의 표준 벤치마크.

- **출처**: OmniAnomaly (Su et al., KDD 2019) — https://github.com/NetManAIOps/OmniAnomaly
- **구성**: 28개 머신(machine-1-1 … 3-11), 38차원, train/test 각 약 2.5만~3만 스텝, 1분 샘플링
- **라이선스**: 저장소 MIT. 데이터 파일만 사용.
- **다운로드**: `tsad-forge download smd` (전체) 또는 `--subset machine-1-1,machine-2-1`
- **로더**: `load_smd(machine="machine-1-1")`

## 알려진 결함

- **이벤트 길이 분산**: 머신별 이상 이벤트 길이가 수 스텝~수천 스텝까지 상이 —
  머신 평균만 보고하면 왜곡됨. 본 저장소는 머신별 결과를 분리 저장한다.
- **준상수 채널**: 일부 채널은 거의 상수 — z-score 정규화 시 0-분산 처리 필요(로더가 처리).
- **train 오염 가능성**: train 구간이 완전 정상이라는 보장은 원 저자 주장에 의존.

## 간단 EDA (로컬 데이터 기준)

라벨된 이상 구간을 음영 처리한 test 채널 샘플과 이벤트 길이 분포. 데이터 다운로드 후 `tsad-forge viz`로 재생성됩니다.

<iframe src="../../../assets/eda/smd.html" width="100%" height="720" frameborder="0"></iframe>
