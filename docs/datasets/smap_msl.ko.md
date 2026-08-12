# NASA SMAP / MSL

**어떤 데이터인가?** NASA 두 미션의 실제 우주선 텔레메트리 — SMAP 위성(토양 수분)과 MSL 큐리오시티 로버. 채널마다 텔레메트리 1개 + 명령 플래그 24개로 구성되고, 이상은 NASA ISA 보고서의 실제 사건이다. *항공우주 텔레메트리* 표준 벤치마크.

- **출처**: NASA telemanom (Hundman et al., KDD 2018) — https://github.com/khundman/telemanom
- **구성**: SMAP 55채널ㆍMSL 27채널, 채널별 train/test npy (25차원: 텔레메트리 1 + command 24)
- **라이선스**: NASA 공개 데이터. telemanom 코드는 사용하지 않음(라벨 csv만 사용).
- **다운로드**: `tsad-forge download smap` — S3(`telemanom/data.zip`) + 라벨 csv.
  S3가 차단된 환경에서는 HuggingFace 미러 `appleparan/telemanom`에서 수동으로 받아
  `data/smap_msl/train/<chan>.npy`, `data/smap_msl/test/<chan>.npy`로 배치하세요.
- **로더**: `load_smap(channel="P-1")`, `load_msl(channel="M-6")`

## 알려진 결함 (숨기지 않음)

- **준이진 채널**: 25차원 중 24개가 command 원핫 계열로 사실상 이진. 실질 정보는
  텔레메트리 1차원에 집중 — 다변량 모델의 이점이 과대평가될 수 있음.
- **희소 라벨**: 채널당 이상 이벤트가 1–3개 수준. event-level 지표의 분산이 큼.
- **PA 부풀림 역사**: 이 데이터셋의 선행 SOTA 보고 다수가 PA-F1 기준 — 본 저장소
  VUS-PR 결과와 비교 불가 (Kim et al., AAAI 2022).