# Yahoo S5 — 신청 안내 (재배포 금지)

**어떤 데이터인가?** Yahoo 서비스의 프로덕션 트래픽 지표(A1: 실데이터, A2–A4: 추세/계절성/변화점을 통제한 합성), 단변량, 시간 단위. 고전 *웹 지표* 벤치마크 — 연구 신청 필요.

Yahoo S5는 **재배포가 금지**되어 있어 이 저장소는 다운로드 스크립트를 제공하지 않습니다.
로컬 배치 시 로더만 제공합니다 (CLAUDE.md §2).

## 신청 방법

1. https://webscope.sandbox.yahoo.com/catalog.php?datatype=s 접속
   ("S5 - A Labeled Anomaly Detection Dataset")
2. Yahoo 계정으로 로그인 후 연구 목적 신청서 제출 (학술 이메일 권장)
3. 승인 메일의 링크로 `ydata-labeled-time-series-anomalies-v1_0.tgz` 다운로드

## 로컬 배치

```
data/yahoo_s5/
├── A1Benchmark/real_1.csv ...      # 실제 트래픽 (67개)
├── A2Benchmark/synthetic_1.csv ... # 합성 (100개)
├── A3Benchmark/ ...                # 합성 + 추세/계절 (100개)
└── A4Benchmark/ ...                # 합성 + changepoint (100개)
```

로더: `load_yahoo(benchmark="A1Benchmark", series="real_1.csv")`

## 알려진 결함

- A2–A4는 합성 — 실데이터 일반화 주장에 사용 금지.
- 공식 train/test 분할이 없어 본 로더는 앞 50%를 train으로 사용(라벨 미사용).
- point 이상 위주 — 긴 collective 이상 평가에는 부적합.