# SWaT / WADI — 신청 안내 (재배포 금지)

**어떤 데이터인가?** SUTD의 물 처리(SWaT, 51채널)·배수(WADI, 123채널) 물리 테스트베드의 센서/액추에이터 로그, 1Hz. 이상은 제어 시스템에 대한 *연출된 사이버-물리 공격*이다. ICS 보안의 기준 벤치마크 — 신청 필요.

SWaT(Secure Water Treatment)와 WADI(Water Distribution)는 싱가포르 SUTD iTrust 소유로
**재배포가 금지**되어 있습니다. 다운로드 스크립트 없이 로더만 제공합니다.

## 신청 방법

1. https://itrust.sutd.edu.sg/itrust-labs_datasets/ 접속
2. "Request access" 양식 제출 (소속·연구 목적 기재; 승인까지 수 일 소요)
3. 승인 후 안내받은 링크에서 다운로드
   - SWaT: `SWaT_Dataset_Normal_v1.xlsx`, `SWaT_Dataset_Attack_v0.xlsx` (2015.12 A1&A2)
   - WADI: `WADI_14days.csv`, `WADI_attackdata.csv` (2017.10)

## 로컬 배치 (xlsx → csv 변환 후)

```
data/swat/train.csv   # Normal 구간 (헤더 정리, 'Normal/Attack' 열 옵션)
data/swat/test.csv    # Attack 구간 ('Normal/Attack' 열 필수)
data/wadi/train.csv   # 14days 정상
data/wadi/test.csv    # attackdata ('Attack' 0/1 열)
```

로더: `load_swat()`, `load_wadi()` — 숫자 열만 사용, 결측 선형 보간.

## 알려진 결함

- **초기 안정화 구간**: SWaT 정상 데이터 앞 ~5–6시간은 플랜트 기동 구간 — 선행 연구
  다수가 제거함. 본 로더는 제거하지 않으므로 필요 시 사용자가 슬라이싱.
- **센서 단위 불일치·상수 채널** 다수(특히 WADI) — 정규화 필수.
- WADI는 버전(2017 vs 2019)에 따라 라벨이 달라 문헌 간 수치 비교가 어렵다.
- 선행 보고 다수가 PA-F1 기준 — 비교 불가.