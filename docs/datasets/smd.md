# SMD (Server Machine Dataset)

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
