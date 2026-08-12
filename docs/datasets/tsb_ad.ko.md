# TSB-AD-U / TSB-AD-M

**어떤 데이터인가?** 메가 벤치마크: 40개 공개 데이터셋(웹 트래픽, 의료, 산업 등)에서 재라벨링·큐레이션한 단변량 1,070개 + 다변량 200개, 고정 train/test 프로토콜 포함. TSAD의 *표준 시험지*에 가장 가깝다.

- **출처**: TheDatumOrg/TSB-AD (Liu & Paparrizos, NeurIPS 2024)
- **구성**: 40개 데이터셋에서 큐레이션한 1070개(U, 단변량) / 200개(M, 다변량) 시계열.
  파일명에 train 길이 인코딩: `..._tr_<N>_1st_<firstAnomaly>.csv`
- **라이선스**: Apache-2.0
- **다운로드**: `tsad-forge download tsb_ad_u` / `tsb_ad_m`
  (호스팅: https://www.thedatum.org/datasets/ — 방화벽 환경에서는 수동 다운로드 후
  `data/tsb_ad_u/`에 압축 해제)
- **로더**: `load_tsb_ad(filename="...", variant="u")`

## 프로토콜 참조 표준

본 저장소는 TSB-AD의 평가 프로토콜(공식 train/test 분할, VUS-PR 주지표)을 참조 표준으로
삼는다 (CLAUDE.md §0). M2의 지표 구현은 TSB-AD 공식 구현과 수치 일치를 테스트로 검증한다.