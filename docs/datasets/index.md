# Datasets

Every loader returns the unified schema `TSADDataset(train[T,D], test[T,D], labels[T], meta)`.
Datasets are **never committed** to this repository — `tsad-forge download <name>` fetches
them into a local `data/` directory, and the SHA256 of every downloaded file is recorded
in a `MANIFEST.json`.

| Dataset | Download | Card |
|---|---|---|
| NASA SMAP/MSL | `tsad-forge download smap` | [smap_msl.md](smap_msl.md) |
| SMD | `tsad-forge download smd` | [smd.md](smd.md) |
| UCR Anomaly Archive | `tsad-forge download ucr` | [ucr.md](ucr.md) |
| TSB-AD-U / TSB-AD-M | `tsad-forge download tsb_ad_u` | [tsb_ad.md](tsb_ad.md) |
| NAB | `tsad-forge download nab` | [nab.md](nab.md) |
| PSM | `tsad-forge download psm` | [psm.md](psm.md) |
| SKAB | `tsad-forge download skab` | [skab.md](skab.md) |
| MGAB | `tsad-forge download mgab` | [mgab.md](mgab.md) |
| MBA (ECG) | `tsad-forge download mba` | [mba.md](mba.md) |
| Yahoo S5 | **application required (no redistribution)** | [yahoo_s5.md](yahoo_s5.md) |
| SWaT / WADI | **application required (no redistribution)** | [swat_wadi.md](swat_wadi.md) |
| SECOM | reference only (not a time series) | [secom.md](secom.md) |

!!! warning "Applies to all datasets"
    Numbers reported in prior papers under the point-adjustment (PA) protocol are
    **not comparable** with the results in this repository. Read the "Known flaws"
    section of each card before interpreting any result.
