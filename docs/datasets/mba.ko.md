# MBA (MIT-BIH Supraventricular Arrhythmia, 심전도)

**어떤 데이터인가?** PhysioNet MIT-BIH 데이터베이스의 2-리드 심전도(ECG)와 심장 전문의의
비트 주석 — 이상은 비정상 비트(부정맥)다. 신호가 준주기적이라 부분수열 기법의 고전적
쇼케이스이며, TSAD 논문 부록에 자주 등장한다(TranAD 등).

- **출처**: https://github.com/imperial-qore/TranAD 의 가공 분할본(`data/MBA`);
  원본: PhysioNet MIT-BIH
- **구성**: 2채널(ECG1/ECG2), train/test 각 약 7.7천 스텝; 라벨은 비트 주석에서 유도
  (비-'N' 비트 주변 ±20 샘플 — TranAD 규약)
- **라이선스**: TranAD 저장소 BSD-3-Clause; 원본 PhysioNet 데이터 ODC-BY (공개)
- **다운로드**: `tsad-forge download mba`
- **로더**: `load_mba()`

## 알려진 결함 / 주의

- **높은 이상 비율(~34%)** — 희소 이상 레짐과 거리가 멀어 정밀도 계열 지표가 다르게
  동작한다. 저비율 데이터셋과 나란히 읽을 것.
- 라벨이 비트 중심 고정 윈도우(±20 샘플)라 정확한 에피소드 경계가 아니다.
- 전체 MIT-BIH가 아닌 TranAD의 소규모 발췌본이다.

## 간단 EDA (로컬 데이터 기준)

라벨된 이상 구간을 음영 처리한 test 채널 샘플과 이벤트 길이 분포. 데이터 다운로드 후 `tsad-forge viz`로 재생성됩니다.

<iframe src="../../../assets/eda/mba.html" width="100%" height="720" frameborder="0"></iframe>
