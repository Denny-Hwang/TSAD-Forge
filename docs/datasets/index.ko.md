# Datasets

모든 로더는 통일 스키마 `TSADDataset(train[T,D], test[T,D], labels[T], meta)`를 반환합니다.
데이터셋은 이 저장소에 **절대 커밋되지 않습니다** — `tsad-forge download <name>`으로 로컬
`data/`에 받고, 받은 파일의 SHA256이 `MANIFEST.json`에 기록됩니다.

| 데이터셋 | 다운로드 | 카드 |
|---|---|---|
| NASA SMAP/MSL | `tsad-forge download smap` | [smap_msl.md](smap_msl.md) |
| SMD | `tsad-forge download smd` | [smd.md](smd.md) |
| UCR Anomaly Archive | `tsad-forge download ucr` | [ucr.md](ucr.md) |
| TSB-AD-U / TSB-AD-M | `tsad-forge download tsb_ad_u` | [tsb_ad.md](tsb_ad.md) |
| NAB | `tsad-forge download nab` | [nab.md](nab.md) |
| PSM | `tsad-forge download psm` | [psm.md](psm.md) |
| SKAB | `tsad-forge download skab` | [skab.md](skab.md) |
| Yahoo S5 | **신청 필요 (재배포 금지)** | [yahoo_s5.md](yahoo_s5.md) |
| SWaT / WADI | **신청 필요 (재배포 금지)** | [swat_wadi.md](swat_wadi.md) |
| SECOM | 참고용 (시계열 아님) | [secom.md](secom.md) |

!!! warning "공통 주의"
    PA(point adjustment) 기반으로 보고된 선행 논문 수치와 이 저장소의 결과는 **직접 비교할 수
    없습니다**. 각 카드의 "알려진 결함" 절을 반드시 읽고 결과를 해석하세요.
